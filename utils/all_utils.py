import joblib
import os 
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap





# Preparation of data
def prepare_data(df):
  X=df.drop("y",axis=1)

  y=df["y"]

  return X,y


def save_model(model,filename): # save model for reuse
  model_dir="models"
  os.makedirs(model_dir,exist_ok=True) # only create if model dir doesnot exists
  filePath=os.path.join(model_dir,filename) #model/filename based onos
  joblib.dump(model,filePath)


#Plot the data for all gates
def save_plot(df,file_name,model):
  def _create_base_plot(df):#internal fun -not available for oustide(underscore bfre fubc)
    df.plot(kind="scatter",x="x1",y="x2",c="y",s=100,cmap="winter")
    plt.axhline(y=0,color="black",linestyle="--",linewidth=1)
    plt.axvline(x=0,color="black",linestyle='--',linewidth=1)
    figure=plt.gcf()#get current figure
    figure.set_size_inches(10,8)

  def _plot_decision_regions(X,y,classifier,resolution=0.02):
    colors=("red","blue","lightgreen","gray","cyan")
    cmap=ListedColormap(colors[: len(np.unique(y))])
    

    X=X.values # values required as array to plot
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1

    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),
                        np.arange(x2_min,x2_max,resolution)) #finds each coordinate value
    print(xx1)
    print(xx1.ravel()) #ravel tries to make it in single array

    z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)

    z=z.reshape(xx1.shape)

    plt.contourf(xx1,xx2,z,alpha=0.2,cmap=cmap) 
    #find xx1,xx2 pass colour bassd on z red or blue region
    #alpha represents transparency

    plt.xlim(xx1.min(),xx1.max())

    plt.ylim(xx2.min(),xx2.max())

    plt.plot()


  X,y=prepare_data(df)
  _create_base_plot(df)
  _plot_decision_regions(X,y,model)

  plot_dir="plots"
  os.makedirs(plot_dir,exist_ok=True) # only create if model dir doesnot exists
  plotPath=os.path.join(plot_dir,file_name) #model/filename based onos
  plt.savefig(plotPath)
