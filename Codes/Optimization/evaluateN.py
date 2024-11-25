#encoding=utf-8


from skimage.metrics import structural_similarity as ssim



def SSIMScore(TruthLabel,PredictImg):
    return ssim(PredictImg,TruthLabel)



def LoadFileForEva(TrueImage,PredictImage):
    return SSIMScore(TrueImage,PredictImage)


    
   