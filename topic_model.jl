using TextAnalysis
input_file = open("/Users/emily/Documents/topic_modeling/MTSamples_Note_Set.txt","r")
lex_file = open("/Users/emily/Documents/topic_modeling/lex_file.txt","w")

# create corpus
sentence_array = readlines(input_file)
sentence_array_n = length(sentence_array)
# remove special characters
for i = 1:sentence_array_n
    sentence_array[i] = replace(sentence_array[i],"["," ")
    sentence_array[i] = replace(sentence_array[i],"]"," ")
    sentence_array[i] = replace(sentence_array[i],"-"," ")
    sentence_array[i] = replace(sentence_array[i],'"'," ")
    sentence_array[i] = replace(sentence_array[i],"'"," ")
    sentence_array[i] = replace(sentence_array[i],"%"," ")
    sentence_array[i] = replace(sentence_array[i],"&"," ")
    sentence_array[i] = replace(sentence_array[i],"#"," ")
    sentence_array[i] = replace(sentence_array[i],"*"," ")
    sentence_array[i] = replace(sentence_array[i],"/"," ")
end
# convert array into corpus
string_document = map(StringDocument, sentence_array)
string_document2 = convert(Array{Any,1}, string_document)

test = TextAnalysis.Corpus(string_document2)

# preprocessing
remove_punctuation!(test)
remove_case!(test)
remove_numbers!(test)
remove_stop_words!(test)

# create lexicon
lexicon(test)
update_lexicon!(test)
lexicon(test)

lex_freq = open("/Users/emily/Documents/topic_modeling/lex_freq.txt", "w")
# sort lexicon in descending order of frequency
for (lex,count) in sort(collect(lexicon(test)), by=tuple -> last(tuple), rev=true)
  write(lex_freq, "$lex\t$count\n")
end
close(lex_freq)

# plot word frequency in the corpus across sentences
using PlotlyJS
data = bar(;x=["patient","history","alcohol","lives","denies","drug","tobacco","married",
        "female","day","smoking","male","white","smoke","illicit"],
        y = [294, 188, 161, 100, 96, 85,84,82,78,76,74,67,65,59,53])
plot(data)

# get keys from the lexicon dictionary
key = keys(lexicon(test))
# create lex file
for (index,value) in enumerate(key)
  write(lex_file, "$index\t$value\n")
end

close(lex_file)

using DataFrames
using TextAnalysis
d = readdlm("/Users/emily/Documents/topic_modeling/MTSamples_Note_Set.txt",'\n')
lex = readtable("/Users/emily/Documents/topic_modeling/lex_file.txt",header = false, separator = '\t')
doc_file = open("/Users/emily/Documents/topic_modeling/doc_file.txt","w")

# rename column names in lex file
rename!(lex,[:x1,:x2],[:term_index,:term])

# calculate word frequency in each sentence
function wordcount(text)
    text = remove_case(text)
    words=split(text,[' ','\n','\t','-','.',',',':',';'];keep=false)
    counts=Dict()
    for w = words
        counts[w]=get(counts,w,0)+1
    end
    return counts
end

# create doc file
for i in 1:size(d)[1]
  freq_df = DataFrame(Any[collect(keys(wordcount(d[i]))), collect(values(wordcount(d[i])))])
  rename!(freq_df,[:x1,:x2],[:term,:frequency])
  # join the lex file with the word frequency within each sentence
  table = join(lex,freq_df,on = :term)
  # extract term index
  index = table[1]
  # extract term frequency within each sentence
  freq = convert(Array{Int64,1}, table[3])
  write(doc_file,"$index\n$freq\n")
end

close(doc_file)

# remove brackets in doc_file and create doc_file2
# run the following command in Terminal:
# sed 's/[^0-9,]//g' doc_file.txt > doc_file2.txt

using TopicModelsVB
srand(25)
corp = readcorp(docfile="/Users/emily/Documents/topic_modeling/doc_file2.txt",
                lexfile="/Users/emily/Documents/topic_modeling/lex_file.txt",
                counts=true)

# filtered LDA model
flda = fLDA(corp, 4)
# setting tol=0.0 ensures all iterations are completed
train!(flda, iter=1000, tol=0.0)

# show top five keywords for each topic, 4 columns displayed on one line in the output
showtopics(flda, 5,cols=4)
