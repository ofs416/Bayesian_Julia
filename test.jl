using DataFrames, CSV, DelimitedFiles
using Plots

X = [readdlm("X.txt", ' ', Float64) ones(1000, 1)] 
y = readdlm("y.txt", ' ', Int64) 

scatter(data[:, 1], data[:, 2], markercolor=labels)

