using LinearAlgebra, StatsBase, HDF5, CSV, DataFrames, JSON, CodecZlib, KNearestCenters, Glob, SHA
using LIBSVM: svmtrain, svmpredict, Kernel
using TextSearch
using BagOfWords
using TextSearch: IdfWeighting, EntropyWeighting, BinaryLocalWeighting, SigmoidPenalizeFewSamples

function as_outfile(testfile, outdir)
    outfile = replace(basename(testfile), ".json" => "", ".gz" => "")
    joinpath(outdir, outfile)
end

as_nick(kwargs) = bytes2hex(sha256(string(kwargs)))

function save_scores(outfile, ypred, yval, scores, meta)
    @info json(scores, 2) 
    mkpath(dirname(outfile))

    h5open(outfile * ".predictions.h5", "w") do f
        f["ypred"] = ypred
        f["yval"] = yval
    end
    
    open(outfile * ".meta.txt", "w") do f
        println(f, string(meta))
    end

    open(outfile * ".predictions.json", "w") do f
        println(f, JSON.json(ypred))
    end
    
    open(outfile * ".scores.json", "w") do f
        println(f, JSON.json(scores, 2))
    end
end


function bow_main(; train, test, outname, nt=Threads.nthreads(),
                    projection=RawVectors(),
                    gw=EntropyWeighting(),
                    lw=BinaryLocalWeighting(),
                    comb=SigmoidPenalizeFewSamples(),
                    qlist=[3, 4, 5],
                    mindocs=3,
                    smooth=0.1,
                    minweight=1e-4,
                    collocations=7,
                    kernel=Kernel.Linear)

    kwargs = (; gw, lw, comb, qlist, mindocs, smooth, minweight, collocations, kernel)
    config = (; projection, kwargs...)
    outname = joinpath(outname, "bow.model_selection=false." * as_nick(config))
    @info "learning model for $outname"
    model = fit(BagOfWordsClassifier, projection, train.text, train.klass; kwargs...)
    @info "predicting $outname"
    y = predict(model, test.text)
    scores = classification_scores(test.klass, y.pred)
    save_scores(outname, y.pred, y.val, scores, (; name="bow", model_selection=false, config))
end

function bow_main(scorefun; train, test, outname, nt=Threads.nthreads(),
                  mindocs_options = [0, 3, 10],
                  gw_options = [EntropyWeighting()],
                  lw_options = [BinaryLocalWeighting()],
                  #projection_options = [RawVectors(), UmapProjection()],
                  projection_options = [RawVectors()],
                  comb_options = [NormalizedEntropy(), SigmoidPenalizeFewSamples()],
                  smooth_options = [0.1],
                  qlist_options = [[3, 4, 5], [2, 5]],
                  collocations_options = [8],
                  kernel_options = [Kernel.Linear],
                  minweight_options = [1e-4], 
    )

    @info "starting model selection"
    config_space = (; gw_options, lw_options,
                      projection_options, comb_options,
                      mindocs_options, smooth_options, minweight_options,
                      qlist_options, collocations_options, kernel_options )

    bestlist = modelselection(scorefun, train.text, train.klass; config_space...)

    @info "=================="
    for b in reverse(bestlist)
        @info b
    end
    
    config = first(bestlist).config
    @info "fitting best model $outname"
    model = fit(BagOfWordsClassifier, train.text, train.klass, config)
    @info "predicting $outname"
    y = predict(model, test.text)
    scores = classification_scores(test.klass, y.pred)
    save_scores(joinpath(outname, "bow.model_selection=true"), y.pred, y.val, scores,
                (; name="bow", model_selection=true, config_list=bestlist, config_space))
end
