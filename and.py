from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model,save_plot
import pandas as pd
import joblib

def main():
    AND={"x1":[0,0,1,1],
     "x2":[0,1,0,1],
     "y":[0,0,0,1],}
    df=pd.DataFrame(AND)
    df
X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model_AND = Perceptron(eta=ETA, epochs=EPOCHS)
model_AND.fit(X, y)

_ = model_AND.total_loss()

save_model(model, "and.model")

loaded_model = joblib.load("models/and.model")
loaded_model.predict(inputs)


save_plot(df, "and.png", model_AND)


if __name__=='__main__':
    main()