my.path <- "d:\\Projects\\Python\\PycharmProjects\\DeepVisualAttentionPrediction\\nowe_eksperymenty\\"

my.dir <- "checkpoint_1pyramids\\"
my.file <- "ALG_1_v1_Page_1.jpg\\results.txt.log"
pp <- paste(my.path, my.dir, my.file, sep = "")
df <- read.csv(pp)
plot(df$epoch, df$loss, type = "l", xlab = "epoch", ylab = "train loss (MSE)")


my.dir <- "checkpoint_2pyramids\\"
pp <- paste(my.path, my.dir, my.file, sep = "")
df <- read.csv(pp)
lines(df$epoch, df$loss, type = "l", col = "red")

my.dir <- "checkpoint_3pyramids\\"
pp <- paste(my.path, my.dir, my.file, sep = "")
df <- read.csv(pp)
lines(df$epoch, df$loss, type = "l", col = "blue")

my.dir <- "checkpoint_4pyramids\\"
pp <- paste(my.path, my.dir, my.file, sep = "")
df <- read.csv(pp)
lines(df$epoch, df$loss, type = "l", col = "cyan")

my.dir <- "checkpoint_places_3pyramids\\"
pp <- paste(my.path, my.dir, my.file, sep = "")
df <- read.csv(pp)
lines(df$epoch, df$loss, type = "l", col = "green")

legend(20, 0.44, legend=c("E-D Pyramid 1", "E-D Pyramid 2","E-D Pyramid 3", "E-D Pyramid 4","2 x E-D Pyramid 1"),
       col=c("black","red", "blue", "cyan","green"), lty = 1, cex=0.8)



my.dir <- "checkpoint_1pyramids\\"
my.file <- "ALG_1_v1_Page_1.jpg\\results.txt.log"
pp <- paste(my.path, my.dir, my.file, sep = "")
df <- read.csv(pp)
plot(df$epoch, df$val_loss, type = "l", xlab = "epoch", ylab = "validation loss (MSE)", ylim = c(0.24, 0.45))


my.dir <- "checkpoint_2pyramids\\"
pp <- paste(my.path, my.dir, my.file, sep = "")
df <- read.csv(pp)
lines(df$epoch, df$val_loss, type = "l", col = "red")

my.dir <- "checkpoint_3pyramids\\"
pp <- paste(my.path, my.dir, my.file, sep = "")
df <- read.csv(pp)
lines(df$epoch, df$val_loss, type = "l", col = "blue")


my.dir <- "checkpoint_4pyramids\\"
pp <- paste(my.path, my.dir, my.file, sep = "")
df <- read.csv(pp)
lines(df$epoch, df$val_loss, type = "l", col = "cyan")

my.dir <- "checkpoint_places_3pyramids\\"
pp <- paste(my.path, my.dir, my.file, sep = "")
df <- read.csv(pp)
lines(df$epoch, df$val_loss, type = "l", col = "green")

legend(20, 0.38, legend=c("E-D Pyramid 1", "E-D Pyramid 2","E-D Pyramid 3","E-D Pyramid 4", "2 x E-D Pyramid 1"),
       col=c("black","red", "blue", "cyan","green"), lty = 1, cex=0.8)


##############################################

my.path <- "d:\\Projects\\Python\\PycharmProjects\\DeepVisualAttentionPrediction\\nowe_eksperymenty\\"
var <- c('ALG_1_v1_Page_1.jpg', 'ALG_1_v2_Page_1.jpg', 'ALG_2_v1_Page_1.jpg', 'ALG_2_v2_Page_1.jpg', 'BIO_Page_1.jpg',
         'FIZ_WB1_Page_1.jpg', 'FIZ_WB2.jpg', 'FIZ_WB3_v1_Page_1.jpg', 'FIZ_WB3_v2_Page_1.jpg', 'FIZ_WB4_stereo_Page_1.jpg',
         'FIZ_WZORY_Page_1.jpg', 'rz 1_Page_1.jpg', 'rz 2_Page_1.jpg', 'rz 3_Page_1.jpg')
my.dir <- "checkpoint_1pyramids\\"
my.file <- "\\results.txt.log"

loss.list <- list()
val_loss.list <- list()

for (a in 1:length(var))
{
  var.var <- paste(my.path, my.dir, var[a], my.file, sep = "")
  df <- read.csv(var.var)
  loss.list[[a]] <- df[10,]$loss
  val_loss.list[[a]] <- df[10,]$val_loss
}

ed1 = unlist(loss.list)
ed1_1 = unlist(val_loss.list)

#round(mean(unlist(loss.list)),3)
#round(sd(unlist(loss.list)),3)
#round(mean(unlist(val_loss.list)),3)
#round(sd(unlist(val_loss.list)),3)

##############################################

my.path <- "d:\\Projects\\Python\\PycharmProjects\\DeepVisualAttentionPrediction\\nowe_eksperymenty\\"
var <- c('ALG_1_v1_Page_1.jpg', 'ALG_1_v2_Page_1.jpg', 'ALG_2_v1_Page_1.jpg', 'ALG_2_v2_Page_1.jpg', 'BIO_Page_1.jpg',
         'FIZ_WB1_Page_1.jpg', 'FIZ_WB2.jpg', 'FIZ_WB3_v1_Page_1.jpg', 'FIZ_WB3_v2_Page_1.jpg', 'FIZ_WB4_stereo_Page_1.jpg',
         'FIZ_WZORY_Page_1.jpg', 'rz 1_Page_1.jpg', 'rz 2_Page_1.jpg', 'rz 3_Page_1.jpg')
my.dir <- "checkpoint_2pyramids\\"
my.file <- "\\results.txt.log"

loss.list <- list()
val_loss.list <- list()

for (a in 1:length(var))
{
  var.var <- paste(my.path, my.dir, var[a], my.file, sep = "")
  df <- read.csv(var.var)
  loss.list[[a]] <- df[10,]$loss
  val_loss.list[[a]] <- df[10,]$val_loss
}

ed2 = unlist(loss.list)
ed2_1 = unlist(val_loss.list)

#round(mean(unlist(loss.list)),3)
#round(sd(unlist(loss.list)),3)

#round(mean(unlist(val_loss.list)),3)
#round(sd(unlist(val_loss.list)),3)

##############################################

my.path <- "d:\\Projects\\Python\\PycharmProjects\\DeepVisualAttentionPrediction\\nowe_eksperymenty\\"
var <- c('ALG_1_v1_Page_1.jpg', 'ALG_1_v2_Page_1.jpg', 'ALG_2_v1_Page_1.jpg', 'ALG_2_v2_Page_1.jpg', 'BIO_Page_1.jpg',
         'FIZ_WB1_Page_1.jpg', 'FIZ_WB2.jpg', 'FIZ_WB3_v1_Page_1.jpg', 'FIZ_WB3_v2_Page_1.jpg', 'FIZ_WB4_stereo_Page_1.jpg',
         'FIZ_WZORY_Page_1.jpg', 'rz 1_Page_1.jpg', 'rz 2_Page_1.jpg', 'rz 3_Page_1.jpg')
my.dir <- "checkpoint_3pyramids\\"
my.file <- "\\results.txt.log"

loss.list <- list()
val_loss.list <- list()

for (a in 1:length(var))
{
  var.var <- paste(my.path, my.dir, var[a], my.file, sep = "")
  print(var.var)
  df <- read.csv(var.var)
  loss.list[[a]] <- df[10,]$loss
  val_loss.list[[a]] <- df[10,]$val_loss
}

ed3 = unlist(loss.list)
ed3_1 = unlist(val_loss.list)
#round(mean(unlist(loss.list)),3)
#round(sd(unlist(loss.list)),3)

#round(mean(unlist(val_loss.list)),3)
#round(sd(unlist(val_loss.list)),3)

##############################################

my.path <- "d:\\Projects\\Python\\PycharmProjects\\DeepVisualAttentionPrediction\\nowe_eksperymenty\\"
var <- c('ALG_1_v1_Page_1.jpg', 'ALG_1_v2_Page_1.jpg', 'ALG_2_v1_Page_1.jpg', 'ALG_2_v2_Page_1.jpg', 'BIO_Page_1.jpg',
         'FIZ_WB1_Page_1.jpg', 'FIZ_WB2.jpg', 'FIZ_WB3_v1_Page_1.jpg', 'FIZ_WB3_v2_Page_1.jpg', 'FIZ_WB4_stereo_Page_1.jpg',
         'FIZ_WZORY_Page_1.jpg', 'rz 1_Page_1.jpg', 'rz 2_Page_1.jpg', 'rz 3_Page_1.jpg')
my.dir <- "checkpoint_4pyramids\\"
my.file <- "\\results.txt.log"

loss.list <- list()
val_loss.list <- list()

for (a in 1:length(var))
{
  var.var <- paste(my.path, my.dir, var[a], my.file, sep = "")
  print(var.var)
  df <- read.csv(var.var)
  loss.list[[a]] <- df[10,]$loss
  val_loss.list[[a]] <- df[10,]$val_loss
}

ed4 = unlist(loss.list)
ed4_1 = unlist(val_loss.list)
#round(mean(unlist(loss.list)),3)
#round(sd(unlist(loss.list)),3)

#round(mean(unlist(val_loss.list)),3)
#round(sd(unlist(val_loss.list)),3)


##############################################

my.path <- "d:\\Projects\\Python\\PycharmProjects\\DeepVisualAttentionPrediction\\nowe_eksperymenty\\"
var <- c('ALG_1_v1_Page_1.jpg', 'ALG_1_v2_Page_1.jpg', 'ALG_2_v1_Page_1.jpg', 'ALG_2_v2_Page_1.jpg', 'BIO_Page_1.jpg',
         'FIZ_WB1_Page_1.jpg', 'FIZ_WB2.jpg', 'FIZ_WB3_v1_Page_1.jpg', 'FIZ_WB3_v2_Page_1.jpg', 'FIZ_WB4_stereo_Page_1.jpg',
         'FIZ_WZORY_Page_1.jpg', 'rz 1_Page_1.jpg', 'rz 2_Page_1.jpg', 'rz 3_Page_1.jpg')
my.dir <- "checkpoint_places_3pyramids\\"
my.file <- "\\results.txt.log"

loss.list <- list()
val_loss.list <- list()

for (a in 1:length(var))
{
  var.var <- paste(my.path, my.dir, var[a], my.file, sep = "")
  df <- read.csv(var.var)
  loss.list[[a]] <- df[10,]$loss
  val_loss.list[[a]] <- df[10,]$val_loss
}

edp = unlist(loss.list)
edp_1 = unlist(val_loss.list)
#round(mean(unlist(loss.list)),3)
#round(sd(unlist(loss.list)),3)


#round(mean(unlist(val_loss.list)),3)
#round(sd(unlist(val_loss.list)),3)

##############################################

sti = c("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14")


df = data.frame(sti, ed1, ed2, ed3, ed4, edp)
sink("D:\\Publikacje\\eye tracking\\Symmetry\\table2wyniki1.txt")
for (a in 1:nrow(df))
{
  pp = paste(df[a,1], "&",
        round(df[a,2],3), "&",
        round(df[a,3],3), "&",
        round(df[a,4],3), "&",
        round(df[a,5],3), "&",
        round(df[a,6],3), "&", "\\\\", sep = "")
  cat(pp)
  cat('\n')
}
pp = paste("Average", "&",
           round(mean(ed1),3), "$\\pm$", round(sd(ed1),3), "&",
           round(mean(ed2),3), "$\\pm$", round(sd(ed2),3), "&",
           round(mean(ed3),3), "$\\pm$", round(sd(ed3),3), "&",
           round(mean(ed4),3), "$\\pm$", round(sd(ed4),3), "&",
           round(mean(edp),3), "$\\pm$", round(sd(edp),3), "\\\\"
           )

cat(pp)
cat('\n')
sink()

######################################

sti = c("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14")


df = data.frame(sti, ed1_1, ed2_1, ed3_1, ed4_1, edp_1)
sink("D:\\Publikacje\\eye tracking\\Symmetry\\table222wyniki1.txt")
for (a in 1:nrow(df))
{
  pp = paste(df[a,1], "&",
             round(df[a,2],3), "&",
             round(df[a,3],3), "&",
             round(df[a,4],3), "&",
             round(df[a,5],3), "&",
             round(df[a,6],3), "&", "\\\\", sep = "")
  cat(pp)
  cat('\n')
}
pp = paste("Average", "&",
           round(mean(ed1_1),3), "$\\pm$", round(sd(ed1_1),3), "&",
           round(mean(ed2_1),3), "$\\pm$", round(sd(ed2_1),3), "&",
           round(mean(ed3_1),3), "$\\pm$", round(sd(ed3_1),3), "&",
           round(mean(ed4_1),3), "$\\pm$", round(sd(ed4_1),3), "&",
           round(mean(edp_1),3), "$\\pm$", round(sd(edp_1),3), "\\\\"
)

cat(pp)
cat('\n')
sink()

######################################
sink("D:\\Publikacje\\eye tracking\\Symmetry\\tablewyniki2v2.txt")
wy = read.table("D:\\Publikacje\\eye tracking\\Symmetry\\wyniki2v2.txt", sep = "\t")
for (a in 1:nrow(wy))
{
  wr = paste(wy[a,1], "&",
             round(wy[a,2],3),"$\\pm$", round(wy[a,3],3), "&",
             round(wy[a,4],3),"$\\pm$", round(wy[a,5],3), "&",
             round(wy[a,6],3),"$\\pm$", round(wy[a,7],3), "&",
             round(wy[a,8],3),"$\\pm$", round(wy[a,9],3), "&",
             round(wy[a,10],3),"$\\pm$", round(wy[a,11],3), "&", round(wy[a,12],0), "\\\\", sep = "")
  cat(wr)
  cat('\n')
}
sink()
