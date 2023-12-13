include("bow.jl")
include("sbert.jl")

function run(; trainfile, testfile, modelname, outdir="data", run_sbert=true, run_bow=true, run_bow_model_selection=true)
    train = read_json_dataframe(trainfile)
    test = read_json_dataframe(testfile)
    outname = as_outfile(testfile, outdir)
    
    if run_sbert && length(glob("$(outname)/sbert*$(basename(modelname))*predictions.json")) == 0
        @info "SBERT $outname"
        sbert_main(; train, test, modelname, outname)
    end
    if run_bow && length(glob("$(outname)/bow.model_selection=false*predictions.json")) == 0
        @info "BOW single config $outname"
        bow_main(; train, test, outname) 
    end
    if run_bow_model_selection && length(glob("$(outname)/bow.model_selection=true*predictions.json")) == 0
        @info "BOW model selection $outname"
        bow_main(; train, test, outname) do y, ypred
            f1_score(y, ypred; weight=:macro)
        end
    end
end


run_huhu2023_es(; kwargs...) = run(;
    trainfile = "datasets/comp2023/IberLEF2023_HUHU_task1_Es_train.json",
    testfile = "datasets/comp2023/IberLEF2023_HUHU_task1_Es_test.json",
    modelname = "hiiamsid/sentence_similarity_spanish_es", #distiluse-base-multilingual-cased-v1"
    kwargs...
)

run_hope2023_es(; kwargs...) = run(;
    trainfile = "datasets/comp2023/IberLEF2023_HOPE_Es_train.json",
    testfile = "datasets/comp2023/IberLEF2023_HOPE_Es_test.json",
    modelname = "hiiamsid/sentence_similarity_spanish_es",
    kwargs...
)
#modelname = "nli-distilroberta-base-v2"
#modelname = "distiluse-base-multilingual-cased-v1"
#modelname = "paraphrase-distilroberta-base-v2"

run_hope2023_en(; kwargs...) = run(;
    trainfile = "datasets/comp2023/IberLEF2023_HOPE_En_train.json",
    testfile = "datasets/comp2023/IberLEF2023_HOPE_En_test.json",
    modelname = "nli-distilroberta-base-v2",
    kwargs...
)

run_hodi2023_it(; kwargs...) = run(;
    trainfile = "datasets/comp2023/evalita2023_HODI_It_train.json",
    testfile = "datasets/comp2023/evalita2023_HODI_It_test.json",
    modelname = "distiluse-base-multilingual-cased-v1",
    kwargs...
)

run_edos2023A_en(; kwargs...) = run(;
    trainfile = "datasets/competitions/semeval2023_task10_A_edos_En_train.json",
    testfile = "datasets/competitions/semeval2023_task10_A_edos_En_test.json",
    modelname = "nli-distilroberta-base-v2",
    kwargs...
)

run_edos2023B_en(; kwargs...) = run(;
    trainfile = "datasets/competitions/semeval2023_task10_B_edos_En_train.json",
    testfile = "datasets/competitions/semeval2023_task10_B_edos_En_test.json",
    modelname = "nli-distilroberta-base-v2",
    kwargs...
)

run_edos2023C_en(; kwargs...) = run(;
    trainfile = "datasets/competitions/semeval2023_task10_C_edos_En_train.json",
    testfile = "datasets/competitions/semeval2023_task10_C_edos_En_test.json",
    modelname = "nli-distilroberta-base-v2",
    kwargs...
)

run_haspeede2023_textual_it(; kwargs...) = run(;
    trainfile = "datasets/comp2023/evalita2023_HaSpeeDe3_It_train.json",
    testfile = "datasets/comp2023/evalita2023_HaSpeeDe3_textual_test.json",
    modelname = "distiluse-base-multilingual-cased-v1",
    kwargs...
)

run_haspeede2023_xreligoushate_it(; kwargs...) = run(;
    trainfile = "datasets/comp2023/evalita2023_HaSpeeDe3_It_train.json",
    testfile = "datasets/comp2023/evalita2023_HaSpeeDe3_XReligiousHate_test.json",
    modelname = "distiluse-base-multilingual-cased-v1",
    kwargs...
)

run_politicit2023_gender_it(; kwargs...) = run(;
    trainfile = "datasets/comp2023/evalita2023_PoliticIT_gender_It_train.json.gz",
    testfile = "datasets/comp2023/evalita2023_PoliticIT_gender_It_test.json.gz",
    modelname = "distiluse-base-multilingual-cased-v1",
    kwargs...
)

run_politicit2023_ideology_binary_it(; kwargs...) = run(;
    trainfile = "datasets/comp2023/evalita2023_PoliticIT_ideology_binary_It_train.json.gz",
    testfile = "datasets/comp2023/evalita2023_PoliticIT_ideology_binary_It_test.json.gz",
    modelname = "distiluse-base-multilingual-cased-v1",
    kwargs...
)

run_politicit2023_ideology_multiclass_it(; kwargs...) = run(;
    trainfile = "datasets/comp2023/evalita2023_PoliticIT_ideology_multiclass_It_train.json.gz",
    testfile = "datasets/comp2023/evalita2023_PoliticIT_ideology_multiclass_It_test.json.gz",
    modelname = "distiluse-base-multilingual-cased-v1",
    kwargs...
)

run_politices2023_gender_es(; kwargs...) = run(;
    trainfile = "datasets/comp2023/IberLEF2023_PoliticEs_gender_Es_train.json.gz",
    testfile = "datasets/comp2023/IberLEF2023_PoliticEs_gender_Es_test.json.gz",
    modelname = "distiluse-base-multilingual-cased-v1",
    kwargs...
)

run_politices2023_profession_es(; kwargs...) = run(;
    trainfile = "datasets/comp2023/IberLEF2023_PoliticEs_profession_Es_train.json.gz",
    testfile = "datasets/comp2023/IberLEF2023_PoliticEs_profession_Es_test.json.gz",
    modelname = "distiluse-base-multilingual-cased-v1",
    kwargs...
)

run_politices2023_ideology_binary_es(; kwargs...) = run(;
    trainfile = "datasets/comp2023/IberLEF2023_PoliticEs_ideology_binary_Es_train.json.gz",
    testfile = "datasets/comp2023/IberLEF2023_PoliticEs_ideology_binary_Es_test.json.gz",
    modelname = "distiluse-base-multilingual-cased-v1",
    kwargs...
)

run_politices2023_ideology_multiclass_es(; kwargs...) = run(;
    trainfile = "datasets/comp2023/IberLEF2023_PoliticEs_ideology_multiclass_Es_train.json.gz",
    testfile = "datasets/comp2023/IberLEF2023_PoliticEs_ideology_multiclass_Es_test.json.gz",
    modelname = "distiluse-base-multilingual-cased-v1",
    kwargs...
)

function main(; kwargs...)
    run_huhu2023_es(; kwargs...)
    run_hope2023_es(; kwargs...)
    run_hope2023_en(; kwargs...)
    run_hodi2023_it(; kwargs...)
    run_edos2023A_en(; kwargs...)
    run_edos2023B_en(; kwargs...)
    run_edos2023C_en(; kwargs...)
    run_haspeede2023_textual_it(; kwargs...)
    run_haspeede2023_xreligoushate_it(; kwargs...)
    run_politicit2023_gender_it(; kwargs...)
    run_politicit2023_ideology_binary_it(; kwargs...)
    run_politicit2023_ideology_multiclass_it(; kwargs...)
    run_politices2023_gender_es(; kwargs...)
    run_politices2023_profession_es(; kwargs...)
    run_politices2023_ideology_binary_es(; kwargs...)
    run_politices2023_ideology_multiclass_es(; kwargs...)
end
