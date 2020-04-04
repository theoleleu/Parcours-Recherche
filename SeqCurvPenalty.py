#Soit model la liste des modèles, data la liste des blocs de donnée et task la tache traitée

def penalty(model,data,task):
    nbtasks=len(model)
    pen=0
    means=[]
    for i in range(nbtasks):
      means.append({})
      for n, p in model[i].named_parameters():
        means[i][n] =p.data
    for i in range(nbtasks):
      fim=fim_diag(model[task],data[i])
      fim=fim[400]
      for n, p in model[task].named_parameters():
        if 'weight' in n: 
          dm=means[task][n]-means[i][n]
          if len(dm.size())>1:
            d=dm[:,0]
          fimd=fim[n]
          while len(fimd.size())>1:
            fimd=fimd[0]
          fimd=fimd[0]
          pen+= fimd*torch.dot(d,d)
    return pen