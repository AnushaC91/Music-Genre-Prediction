library(gam)
library(stats)
Data <- read.table("clusteringData.csv",header = TRUE, sep=",",stringsAsFactors = FALSE)
str(Data)
na.fail(Data)
unique (unlist (lapply (Data, function (x) which (is.na (x)))))
#na.gam.replace(Data)
#na.exclude(Data)
Data <- na.omit(Data)
X <- Data[,1:15]
y <- Data$genre
MusicG_clusters <- kmeans(X,4)
X$cluster_id <- MusicG_clusters$cluster
Data$cluster <- MusicG_clusters$cluster
Genre_CID <- Data[c("cluster","genre")]
IDs<-table(Genre_CID$cluster,Genre_CID$genre)
ID <- data.frame(IDs)
k <- c(2,3,4,5,6,7,8)
WithinSS <- c()
BetweenSS <- c()
No_clusters <- c()
for (i in 1:length(k)){
      MusicG_clusters1 <- kmeans(X,k[i])
	WithinSS[i] <- MusicG_clusters1$tot.withinss
	BetweenSS[i] <- MusicG_clusters1$betweenss
	No_clusters[i]<- k[i]
}
par(pch=22, col="red") # plotting symbol and color 
par(mfrow=c(1,2)) # all plots on one page 
plot(No_clusters, WithinSS, type="o", main="WithinHomegenity") 
plot(No_clusters, BetweenSS,  type="o", main="BetweenHeterogenity") 
