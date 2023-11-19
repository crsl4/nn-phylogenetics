#!/bin/bash

#convert sample data to a fasta file as input data
datafile = "test.in"
lines = readlines(datafile)
fastafile = "test.fasta"
io = open(fastafile, "w")

n = length(lines)
l = length(lines[1])

write(io,"$n $l \n")
for i in 1:n
   write(io, string(">",i,"\n"))
   write(io, lines[i])
   write(io, "\n")
end

close(io)
