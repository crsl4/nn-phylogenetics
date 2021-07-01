library(plyr)
options(digits = 4)
input <- read.csv(file = 'output.csv',header = FALSE)

njdist <- input[which(input$V1 == 'njdist'),]
sumnj <- count(njdist,"V2")
totalnj<-sum(sumnj[,'freq'])
cornj<-sumnj[which(sumnj$V2 == 0),]
corrnj<-cornj$freq
accnj <-corrnj/totalnj

mldist <- input[which(input$V1 == 'mldist'),]
summl <- count(mldist,"V2")
totalml<-sum(summl[,'freq'])
corml<-summl[which(summl$V2 == 0),]
corrml<-corml$freq
accml <-corrml/totalml

write.table(accnj,file="accuracyresult.csv",append = T,sep=',',row.names="AccofNJ",col.names=F)
write.table(accml,file="accuracyresult.csv",append = T,sep=',',row.names="AccofML",col.names=F)
