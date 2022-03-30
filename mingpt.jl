using ParameterSchedulers
using Flux
using LinearAlgebra
using Random
using Zygote
using BSON: @save
using BSON: @load
using CUDA
using Printf

ndigit = 2


struct CausalSelfAttention
  key::Dense
  query::Dense
  value::Dense
  attn_drop::Dropout
  resid_drop::Dropout
  proj::Dense
  n_head::Int64
#   mask::Any
end
function CausalSelfAttention(config::Dict)
  n = config["n_embed"]
  key = Dense(n, n)
  query = Dense(n, n)
  value = Dense(n, n)
  # regularization
  attn_drop = Dropout(config["attn_pdrop"])
  resid_drop = Dropout(config["resid_pdrop"])
  # output projection
  proj = Dense(n, n)

CausalSelfAttention(key, query, value, attn_drop, resid_drop, proj, config["n_head"])
end


function (m::CausalSelfAttention)(x)
  C, T, B = size(x)

  perhead = Int64(C ÷ m.n_head)

  kj = m.key(x)
  kj = reshape(kj, m.n_head, perhead, T, B)
  kj = permutedims(kj, (1, 3, 2, 4))

  qj = m.query(x)
  qj = reshape(qj, m.n_head, perhead, T, B)
  qj = permutedims(qj, (1, 3, 2, 4))

  vj = m.value(x)
  vj = reshape(vj, m.n_head, perhead, T, B)
  vj = permutedims(vj, (1, 3, 2, 4))

  att1j = reshape(
    batched_mul(batched_transpose(reshape(kj, m.n_head, T, :)), reshape(qj, m.n_head, T, :)) ./
    Float32(sqrt(size(kj, 1))),
    (T, T, perhead, B),
  )

  mask =
    (1 .- reshape(triu(ones(Float32, T, T)), Val(4))) .* (-1f9) |>
    gpu
  att1j = att1j .+ mask
  att1j = Flux.softmax(att1j)
  att1j = m.attn_drop(att1j)

  y1j = reshape(
    batched_mul(reshape(vj, m.n_head, T, :), reshape(att1j, T, T, :)),
    (m.n_head, T, perhead, B),
  )
  y1j = reshape(permutedims(y1j, (1, 3, 2, 4)), :, T, B)
  y1j = m.proj(y1j)
  y1j = m.resid_drop(y1j)
end

Flux.@functor CausalSelfAttention

Flux.trainable(a::CausalSelfAttention) = (
  a.key.weight,
  a.key.bias,
  a.query.weight,
  a.query.bias,
  a.value.weight,
  a.value.bias,
  a.proj.weight,
  a.proj.bias,
)
decayed_trainable(a::CausalSelfAttention) =
  (a.key.weight, a.query.weight, a.value.weight, a.proj.weight)


struct Block
  ln1::LayerNorm
  ln2::LayerNorm
  attn::CausalSelfAttention
  d1::Dense
  d2::Dense
  drop::Dropout
  mlp::Chain
end

function Block(config::Dict)
  n = config["n_embed"]
  ln1 = LayerNorm(n)
  ln2 = LayerNorm(n)

  attn = CausalSelfAttention(config)

  d1 = Dense(n, 4 * n, gelu)
  d2 = Dense(4 * n, n)
  drop = Dropout(config["resid_pdrop"])
  mlp = Chain(d1, d2, drop)
  Block(ln1, ln2, attn, d1, d2, drop, mlp)
end

function (m::Block)(x)
  out = m.ln1(x)
  out = m.attn(out)
  xplus = x .+ out

  out = m.ln2(xplus)
  out = m.mlp(out)
  return xplus .+ out
end

Flux.@functor Block
Flux.trainable(a::Block) = (
  a.ln1.diag.α,
  a.ln1.diag.β,
  a.ln2.diag.α,
  a.ln2.diag.β,
  Flux.trainable(a.attn)...,
  a.d1.weight,
  a.d1.bias,
  a.d2.weight,
  a.d2.bias,
)
decayed_trainable(a::Block) = (decayed_trainable(a.attn)..., a.d1.weight, a.d2.weight)


struct Gpt
  tok_emb::Flux.Embedding
  pos_emb::Array{Float32,3}
  dropout::Dropout
  blocks::Chain
  ln::LayerNorm
  head::Dense
end


function Gpt(config::Dict)
  tok_emb = Flux.Embedding(config["vocab_size"], config["n_embed"])
  pos_emb = zeros(Float32, (config["n_embed"], config["block_size"], 1))
  dropout = Dropout(config["embd_pdrop"])
  blocks = Chain([Block(config) for i = 1:config["n_layer"]]...)
  ln = LayerNorm(config["n_embed"])
  head = Dense(config["n_embed"], config["vocab_size"], bias = false)
  Gpt(tok_emb, pos_emb, dropout, blocks, ln, head)
end

function (m::Gpt)(idx)
  t, b = size(idx)
  token_embeddings = m.tok_emb(idx)
  position_embeddings = m.pos_emb[:, 1:t, :] |> gpu
  x = m.dropout(token_embeddings .+ position_embeddings)
  x = m.blocks(x)
  x = m.ln(x)
  return m.head(x)
end

Flux.@functor Gpt
function Flux.trainable(a::Gpt)
  p = []
  for b in a.blocks
    for e in Flux.trainable(b)
      push!(p, e)
    end
  end
  push!(p, a.tok_emb.weight, a.pos_emb, a.ln.diag.α, a.ln.diag.β, a.head.weight, a.head.bias)
    # push!(p, a.tok_emb.weight, a.ln.diag.α, a.ln.diag.β, a.head.weight, a.head.bias)
  return Tuple(p)
end

function decayed_trainable(a::Gpt)
  p = []
  for b in a.blocks
    for e in decayed_trainable(b)
      push!(p, e)
    end
  end
  push!(p, a.head.weight)
  return Tuple(p)
end


function dataItem(ix, ndigit = 2)
  nd = 10^ndigit
  a = ix ÷ nd
  b = ix % nd
  c = a + b

  render = lpad(a, 2, "0") * lpad(b, 2, "0") * lpad(c, 2 + 1, "0")

  dix = [parse(Int, i) for i in render]
  x = dix[1:end-1]
  y = dix[2:end]
  y[1:(ndigit*2-1)] .= -100
  x, y
end


function makeData(ndigit = 2, test_pct = 0.2)
  num_all = (10^ndigit)^2
  all_data_idx = shuffle(0:num_all-1)
  test_end_idx = (Int)(num_all * test_pct)
  test_idx = all_data_idx[1:test_end_idx]
  train_idx = all_data_idx[test_end_idx+1:end]

  test_data_x = Array{Int32,2}(undef, ndigit * 3, test_end_idx)
  test_data_y = Array{Int32,2}(undef, ndigit * 3, test_end_idx)
  for (i, ix) in enumerate(test_idx)
    x, y = dataItem(ix, ndigit)
    test_data_x[:, i] = x
    test_data_y[:, i] = y
  end

  train_data_x = Array{Int32,2}(undef, ndigit * 3, num_all - test_end_idx)
  train_data_y = Array{Int32,2}(undef, ndigit * 3, num_all - test_end_idx)
  for (i, ix) in enumerate(train_idx)
    x, y = dataItem(ix, ndigit)
    train_data_x[:, i] = x
    train_data_y[:, i] = y
  end
  return (train_data_x, train_data_y), (test_data_x, test_data_y)
end

function addOneForJulia(data)
  data .= data .+ 1
end


mutable struct MYADAM
  eta::Float64
  beta::Tuple{Float64,Float64}
  epsilon::Float64
  wd::Float64
  state::IdDict
  sched_mult::Real
end

MYADAM(η::Real, β::Tuple, wd::Real, state::IdDict) = MYADAM(η, β, Flux.Optimise.EPS, wd, state, 1f0)

function MYADAM(
  allParams,
  decayedParams,
  η::Real = 0.001f0,
  β::Tuple = (0.9f0, 0.999f0),
  wd::Real = 0.1f0,
)
  myiddict = IdDict()
  for p in allParams
    myiddict[p] = (zero(p), zero(p), Float64[β[1], β[2]], p in decayedParams ? true : false)
  end
  MYADAM(η, β, wd, myiddict)
end

function Flux.Optimise.apply!(o::MYADAM, x, Δ)
    η, β = o.eta, o.beta
    wd = o.wd

    mt, vt, βp, decay = get(o.state, x, (zero(x), zero(x), Float64[β[1], β[2]], false))

    ## Clip norm
    Δnrm = norm(Δ)
    if Δnrm > 1.0f0
        rmul!(Δ, 1.0f0 / Δnrm)
    end


    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
    @. Δ = o.sched_mult * (mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon))
    βp .= βp .* β

    if decay
        @. Δ += o.sched_mult * wd * x
    end

    ## Descent
    Δ .*= o.eta

  return Δ
end

function myloss(gpt, x, allRowsButLast, tot)
  logits = gpt(x)
  loss = -sum(sum(allRowsButLast .* Flux.Losses.logsoftmax(logits, dims = 1), dims = 1)) / tot
  return loss
end


function mytraining(trnx, trny, tstx, tsty, config)
  gpt = Gpt(config) |> gpu
  allParams = Flux.params(gpt)
  decayedParams = Flux.Params()
  push!(decayedParams, decayed_trainable(gpt)...)

  lr = config["learning_rate"]
  opt = MYADAM(allParams, decayedParams, lr, config["betas"])
    # opt = ADAM(config[learning_rate], config[betas])
    # opt = Flux.Optimiser(adam, WeightDecay(0.1))

  train_loader =
    Flux.DataLoader((data = trnx, label = trny), batchsize = config["batch_size"], shuffle = true)
  test_loader =
    Flux.DataLoader((data = tstx, label = tsty), batchsize = config["batch_size"], shuffle = true)

  labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -99]
  tokens = 0
  best_loss = Inf32
  for epoch = 1:config["max_epochs"]


    for (it, (x, y)) in enumerate(train_loader)
      x = x |> gpu

      allRowsButLast =
        Flux.onehotbatch(y, labels)[1:end-1, :, :] |> gpu

      total_num_of_entries = reduce(*, size(y))

      tot = total_num_of_entries - count(i -> i == -99, y) #|> gpu

      train_loss, back = Zygote.pullback(() -> myloss(gpt, x, allRowsButLast, tot), allParams)

      gs = back(one(train_loss))

      Flux.Optimise.update!(opt, allParams, gs)

      if config["lr_decay"]
        tokens += count(i -> i >= 0, y)
        if tokens < config["warmup_tokens"]
          lr_mult = Float32(tokens / max(1, config["warmup_tokens"]))
        else
          progress = Float32(
            (tokens - config["warmup_tokens"]) / max(1, config["final_tokens"] - config["warmup_tokens"]),
          )
          lr_mult = max(0.1f0, 0.5f0 * (1.0 + cos(pi * progress)))
        end
        # lr = config[learning_rate] * lr_mult
        opt.sched_mult = lr_mult

      end

      if it % 10 == 1
        @printf "Epoch: %d Iter: %d Train Loss: %.2f lr_mult: %.2f tokens: %d\n" epoch it train_loss lr_mult tokens
        # println(
        #   "Epoch: $(epoch) Iter: $(it) Train Loss: $(train_loss) lr: $(lr) lr_mult: $(lr_mult) tokens: $(tokens)",
        # )
      end
    end

    test_loss = epochOnTest(test_loader, gpt, labels)
    if test_loss < best_loss
      best_loss = test_loss
      @save "mymodel.bson" gpt
    end
  end
  return gpt
end


function epochOnTest(test_loader, model, labels)
  losses = Vector{Float32}(undef, 0)
  for (it, (x, y)) in enumerate(test_loader)
    x = x |> gpu
    allRowsButLast = Flux.onehotbatch(y, labels)[1:end-1, :, :] |> gpu
    total_num_of_entries = reduce(*, size(y))
    tot = Float32(total_num_of_entries - count(i -> i == -99, y))
    loss = myloss(model, x, allRowsButLast, tot)
    push!(losses, loss)
  end
  test_loss = Flux.mean(losses)
  println("Test Loss: $(test_loss)")
  return test_loss
end

function sample(config, model, x, steps, temperature = 1.0, sample = false, top_k = nothing)
  bs = config["block_size"]

  for k = 1:steps
    S, B = size(x)
    x_cond = nothing
    if S <= bs
      x_cond = x
    else
      x_cond = x[end-bs:end, :]
    end
    x_cond = x_cond |> gpu
    logits = model(x_cond)
    logits = logits[:, end, :] / temperature

    probs = Flux.softmax(logits)

    ix = getindex.(argmax(probs, dims = 1), 1)
    x = vcat(x, ix)
  end
  return x
end


function give_exam(gpt, x, y, config)
  data_loader =
    Flux.DataLoader((data = x, label = y), batchsize = config["batch_size"], shuffle = true)

  tot_correct = 0
  tot = 0
  for (x, y) in data_loader

    d1d2 = x[1:ndigit*2, :]
    d1d2d3 = sample(config, gpt, d1d2, ndigit + 1)
    d3 = d1d2d3[end-(ndigit):end, :]

    factors = [10^i for i = ndigit:-1:0]
    d1i = sum((d1d2.-1)[1:ndigit, :] .* factors[2:end], dims = 1)
    d2i = sum((d1d2.-1)[ndigit+1:ndigit*2, :] .* factors[2:end], dims = 1)
    d3i_pred = sum((d3 .- 1) .* factors, dims = 1)
    d3_gt = d1i .+ d2i
    correct = sum(d3_gt .== d3i_pred)
    tot += size(x, 2)
    tot_correct += correct
  end
  println("tot: $(tot) tot_correct: $(tot_correct)")
end
