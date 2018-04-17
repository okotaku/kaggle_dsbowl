import pandas as pd

df = pd.read_table("result.txt", header=None)
i = 0
result = []
for col in df.values:
    if "val_loss" in col[0]:
        i += 1
        trainloss = col[0].split("loss: ")[1].split(" - rpn_class")[0]
        valloss = col[0].split("val_loss: ")[1].split(" - val_rpn_class_loss:")[0]
        result.append([i, trainloss, valloss])
        
result_df = pd.DataFrame(result)
result_df[result_df.iloc[:, 2] == min(result_df.iloc[:, 2])]