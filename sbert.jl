using PyCall

pushfirst!(pyimport("sys")."path", "")
sbert = pyimport("sbertencoder")


function sbert_encode(E, text; fuse=:mean)
    X = E.encode(text)

    if size(X, 1) == 1
        normalize!(vec(X))
    else
        if fuse === :mean
            mean(X, dims=1) |> vec |> normalize!
        else
            throw(ArgumentError("Unknown fusion method $fuse; known methods = [:mean]"))
        end
    end
end

function sbert_encode_xcorpus(E, corpus)
    X = E.encode(corpus) |> permutedims
    
    for c in eachcol(X)
        normalize!(c)
    end

    X
end

function sbert_encode_corpus(E, corpus)
    x = sbert_encode(E, corpus[1])
    X = Matrix{Float32}(undef, length(x), length(corpus))
    X[:, 1] .= x

    for i in 2:length(corpus)
        X[:, i] .= sbert_encode(E, corpus[i])
    end
    
    X
end

function sbert_create_model(text, labels, E; kernel=Kernel.Linear, nt=Threads.nthreads())
    Xtrain = sbert_encode_corpus(E, text)
    @show size(Xtrain) length(labels)

    weights = let C = countmap(labels)
       s = sum(values(C))
       nc = length(C)
       Dict(label => s/(nc * count) for (label, count) in C)
    end

    svmtrain(Xtrain, labels; kernel, nt, weights), Xtrain
end

function sbert_main(; train, test, modelname, outname, kernel=Kernel.Linear, nt=Threads.nthreads())
    modelnick = basename(modelname)
    E = sbert.TextEncoder(modelname, modelnick)
    model, Xtrain = sbert_create_model(train.text, train.klass, E; nt, kernel)
    Xtest = sbert_encode_corpus(E, test.text)
    @info "predicting $outname"
    ypred, yval = svmpredict(model, Xtest; nt)
    scores = classification_scores(test.klass, ypred)
    outfile = joinpath(outname, "sbert-$modelnick")

    save_scores(outfile, ypred, yval, scores, (; name="sbert", config=(; kernel, modelname)))  
    
    h5open(outfile * ".h5", "w") do f
        f["Xtrain"] = Xtrain
        f["ytrain"] = train.klass
        f["Xtest"] = Xtest
        f["ytest"] = test.klass
    end


end


