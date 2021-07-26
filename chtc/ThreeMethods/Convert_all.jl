datafile = "test0.in"
lines = readlines(datafile)
fastafile = "test0.fasta"
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
datafile = "test1.in"
lines = readlines(datafile)
fastafile = "test1.fasta"
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
datafile = "test2.in"
lines = readlines(datafile)
fastafile = "test2.fasta"
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
datafile = "test3.in"
lines = readlines(datafile)
fastafile = "test3.fasta"
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
datafile = "test4.in"
lines = readlines(datafile)
fastafile = "test4.fasta"
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

datafile = "test5.in"
lines = readlines(datafile)
fastafile = "test5.fasta"
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

datafile = "test6.in"
lines = readlines(datafile)
fastafile = "test6.fasta"
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

datafile = "test7.in"
lines = readlines(datafile)
fastafile = "test7.fasta"
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

datafile = "test8.in"
lines = readlines(datafile)
fastafile = "test8.fasta"
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

datafile = "test9.in"
lines = readlines(datafile)
fastafile = "test9.fasta"
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
