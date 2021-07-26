library(plyr)
options(digits = 4)
input <- read.csv(file = 'output.csv',header = FALSE)

njdist <- input[which(input$V1 == 'njdist'),]
sumnj <- count(njdist,"V2")
totalnj<-sum(sumnj[,'freq'])
cornj<-sumnj[which(sumnj$V2 == 0),]
corrnj<-cornj$freq
accnj <-corrnj/totalnj
if(identical(accnj,numeric(0))){
accnj<-0
}

mldist <- input[which(input$V1 == 'mldist'),]
summl <- count(mldist,"V2")
totalml<-sum(summl[,'freq'])
corml<-summl[which(summl$V2 == 0),]
corrml<-corml$freq
accml <-corrml/totalml
if(identical(accml,numeric(0))){
accml<-0
}

bidist <- input[which(input$V1 == 'bidist'),]
sumbi <- count(bidist,"V2")
totalbi<-sum(sumbi[,'freq'])
corbi<-sumbi[which(sumbi$V2 == 0),]
corrbi<-corbi$freq
accbi <-corrbi/totalbi
if(identical(accbi,numeric(0))){
accbi<-0
}

write.table(accnj,file="accuracyresult.csv",append = T,sep=',',row.names=1,col.names=F)
write.table(accml,file="accuracyresult.csv",append = T,sep=',',row.names=2,col.names=F)
write.table(accbi,file="accuracyresult.csv",append = T,sep=',',row.names=3,col.names=F)