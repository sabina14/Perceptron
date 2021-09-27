from os import closerange
import numpy as np

class Perceptron:
  
  def __init__(self,eta,epoochs):
    self.weights=np.random.randn(3)*1e-4 #small weight initialization
    print("intial weights before training:\n {}".format(self.weights))
    self.eta=eta #learning rate
    self.epoochs=epoochs

  def activationfunction(self,inputs,weights):
    z=np.dot(inputs,weights)
    return np.where(z>0,1,0)
  
  def fit(self,X,y):
    self.X=X #self dot X for to use anywhere in the class
    self.y=y

    X_with_bias=np.c_[self.X,-np.ones((len(self.X),1))] #c is concat
    #X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
    print("X with bias: \n{}".format(X_with_bias))


    for epooch in range(self.epoochs):
      print("--"*10)#seperator
      print(f"for epooch:{epooch}")
      print("--"*10)

      yhat=self.activationfunction(X_with_bias,self.weights) #forward pass
      print("Predicted value after fprward pass:\n{}".format(yhat))

      self.error=self.y-yhat
      print("Errors:\n {}".format(self.error))

      self.weights=self.weights + self.eta *  np.dot(X_with_bias.T,self.error) #backward propagation
      print(f"update weights after epoch:\n{epooch}/{self.epoochs} : {self.weights}")
      print("####"*5)



  def predict(self,X):
     X_with_bias=np.c_[X,-np.ones((len(X),1))]
     return self.activationfunction(X_with_bias,self.weights)
    
  def total_loss(self):
    total_loss=np.sum(self.error)
    print(f"Total loss(){total_loss}")
     #f automtaicaaly to string no need for conversion for float values too
    return total_loss
