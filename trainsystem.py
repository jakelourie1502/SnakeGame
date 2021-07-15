
def fit(dataloaders, model, target_model,optimizer, criterion, metrics, patience, epochs, device, gamma):
    import torch    
    def ModelOutputAndMetrics(batch, model, target_model, metrics, device, gamma, TTV='train'):
        '''
        In: dataloaders, the model, the metrics dict, the device & whether it's train or val
        Out: creates output, and prints metric --> can pass output to loss/optimizer for train
        '''

        image, next_image, move, reward, done = batch
        image, next_image, move, reward, done = image.float(), next_image.float(), move.float(), reward.float(), done.float()
        s = len(image)
        #getting qsa
        Qvalues = model(image)  
        Qsa = Qvalues[range(s),move.long()].float()
        
        #getting targets
        QvaluesNext = target_model(next_image)
        best_actions = torch.argmax(QvaluesNext,dim=1)
        QSA_next = QvaluesNext[range(s),best_actions].float()
        target=reward+(1-done)*gamma*QSA_next
        target = target
        for metric in metrics.values():
            metric(Qsa,target)
        return Qsa, target

    def TrainEpoch(dataloaders, model, target_model, optimizer, criterion, metrics,device, gamma):
        model.train()
        for batch in dataloaders['train']:
            Qsa, target = ModelOutputAndMetrics(batch, model, target_model, metrics, device, gamma, 'train')
            loss = criterion(Qsa, target)
            loss.backward()
            optimizer.step(); optimizer.zero_grad()
        for MetricName, metric in metrics.items():
            met = metric.compute()
            metric.reset() 

    def ValEpoch(dataloaders, model, target_model, optimizer, criterion, metrics,device, gamma):
        model.eval()
        with torch.no_grad():
            for batch in dataloaders['val']:
                Qsa, target = ModelOutputAndMetrics(batch, model, target_model, metrics, device, gamma, 'val')
            for MetricName, metric in metrics.items():
                met = metric.compute()
                if MetricName == 'MSE':
                    MSE = met
                print(MetricName,': ', met.item())
                metric.reset() 
                return met

    MSE = float('inf')
    patience_c =0    
    for epoch in range(epochs):
      TrainEpoch(dataloaders, model, target_model, optimizer, criterion, metrics, device, gamma)
      if (epoch) % 2 == 0:
        print(f'Epoch {epoch} Val Metrics: ')
        trial_MSE = ValEpoch(dataloaders, model, target_model, optimizer, criterion, metrics, device, gamma)
        if trial_MSE < MSE:
          MSE = trial_MSE
          patience_c=0
        else:
          patience_c+=1
          if patience_c > patience and epoch > 0 :
            print('early stop')
            break                    
    

