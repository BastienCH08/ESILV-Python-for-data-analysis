# app.py
from flask import Flask, render_template
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns

import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os

from mapping import *
# Import the dataframe
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Load your dataset
df = pd.read_csv("ObesityDataSet.csv")

# ... (rest of the processing code)
# rename the columns
df.columns=["Gender","Age","Height","Weight",'Family_Overweight',
    "High_Caloric", "Vegetables","Daily_Main_Meals",
    "Food_Between_Meals","Smoke",'Water','Monitor_Calories',
    'Physical_Activity','Screen_Time','Alcohol','Transport','Obesity']

# round the values and convert into the right type
df["Age"] = df["Age"].astype(int)
df["Daily_Main_Meals"]= df["Daily_Main_Meals"].round(1)
df[["Weight","Height"]] = df[["Weight","Height"]].round(2)
columns_to_round=["Vegetables", "Physical_Activity","Daily_Main_Meals","Screen_Time", "Water"]
df[columns_to_round]= df[columns_to_round].round(0).astype(int)

# compute Mass Body Index
df["MBI"] = (df["Weight"]/(df["Height"]**2)).round(2)


# map the answers
df["Screen_Time"] = df["Screen_Time"].map(Screen_Time)
df["Transport"]= df["Transport"].map(MTRANS)
df["Vegetables"] = df["Vegetables"].map(Vegetables)
df["Daily_Main_Meals"] = df["Daily_Main_Meals"].map(DailyMainMeals)
df["Water"] = df["Water"].map(Water)
df["Physical_Activity"] = df["Physical_Activity"].map(Physical_Activity)


# Creating a instance of label Encoder.
le = LabelEncoder()

# Using .fit_transform function to fit label
# encoder and return encoded label
label = df.select_dtypes(include=["object"]).apply(lambda col : le.fit_transform(col))

# printing label
df_numeric =  pd.concat([label.reset_index(drop=True), df.select_dtypes(exclude=["object"])], axis=1) 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_numeric.drop(["Obesity","MBI"], axis=1), df_numeric[["Obesity"]], test_size=0.33)

# Standardize the values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train) # Il ne faut fiter que sur les data d'entrainement
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)  # apply same transformation to test data



app = Flask(__name__)


@app.route('/')
def home():
    
    # Convert the DataFrame to HTML
    table_html = df.to_html(classes='table', index=False)

    return render_template('home.html', table_html=table_html)

@app.route('/notebook')
def notebook():
    return render_template('notebook.html')

@app.route('/miscellaneous')
def miscellaneous():
    return render_template('miscellaneous.html')


@app.route('/plot')
def plot():
    fig = create_figure()
    fig.savefig("static/graphs.png")
    return render_template("plot.html", user_image="static/test.png")

def create_figure():
    fig = Figure(figsize=(19.2, 10.8), dpi=100)
    # First subplot (pie chart)
    desired_weight_order = [
        'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
        'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
    ]
    array = df["Obesity"].value_counts()[desired_weight_order]

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.pie(array, startangle=0, shadow=True, autopct='%1.0f%%', pctdistance=0.6, textprops={'weight': 'bold', 'size': 14},
            explode=[0.05 for _ in range(len(array))])
    ax1.legend(desired_weight_order, bbox_to_anchor=(1.5, 0.5), loc='right', frameon=False)
    ax1.set_title("OBESITY REPARTITION", fontsize=16, fontweight='bold')

    # Second subplot (scatter plot)
    desired_legend_order = [
        'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
        'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
    ]

    # Create a custom color palette for the obesity levels
    custom_palette = sns.color_palette("rainbow", n_colors=len(desired_legend_order))

    ax2 = fig.add_subplot(2, 2, 2)
    scatter = sns.scatterplot(data=df, x="Age", y="MBI", hue="Obesity", palette=custom_palette, marker="o", s=20,
                              linewidth=0, ax=ax2)

    # Get the current legend
    legend = scatter.get_legend()

    # Create a custom legend in the desired order
    handles, labels = [], []
    for level in desired_legend_order:
        index = df["Obesity"].unique().tolist().index(level)
        handles.append(legend.legendHandles[index])
        labels.append(level)

    legend.remove()  # Remove the default legend
    ax2.legend(handles, labels, title_fontsize=12, loc='upper right', frameon=False)
    ax2.tick_params(left=False, right=False, labelleft=False)
    ax2.set_title("AGE vs. MBI BY OBESITY", fontsize=16, fontweight='bold')

    # Third subplot (bar chart for GENDER vs. OBESITY)
    ax3 = fig.add_subplot(2, 2,  3)

    crosstab = pd.crosstab(df['Gender'], df['Obesity'])
    crosstab = crosstab[desired_weight_order]
    crosstab.plot(kind='bar', stacked=True, colormap='viridis', ax=ax3)
    ax3.set_title("GENDER vs. OBESITY", fontsize=16, fontweight='bold')
    ax3.set_xlabel("")
    ax3.tick_params(left=False, right=False, labelleft=False, bottom=False)
    ax3.legend(title='Obesity', bbox_to_anchor=(1.05, .75), loc='upper left', frameon=False)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

    # Fourth subplot (bar chart for OBESITY TYPES by PHYSICAL ACTIVITY)
    ax4 = fig.add_subplot(2, 2, 4)

    df_counts = df.groupby(['Physical_Activity', 'Obesity']).size().unstack()
    df_percentage = df_counts.div(df_counts.sum(axis=1), axis=0) * 100
    df_percentage = df_percentage[desired_weight_order]

    df_percentage.plot(kind='bar', stacked=True, cmap='viridis', ax=ax4)

    ax4.set_title("OBESITY TYPES by PHYSICAL ACTIVITY", fontsize=16, fontweight='bold')
    ax4.set_xlabel("Physical Activity Level")
    ax4.set_ylabel("Percentage")
    ax4.tick_params(left=False, bottom=False, labelleft=False)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
    ax4.legend(title="Obesity Type", bbox_to_anchor=(1.05, 0.7), frameon=False)

    fig.tight_layout()

    return fig



def create_figure_predictions():
    fig = Figure(figsize=(19.2, 6.8), dpi=100)
    axes = fig.subplots(nrows=1, ncols=3)
    fig.suptitle("MODEL COMPARISON", fontsize=16, fontweight='bold')

    models = {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": GaussianNB(),
    }

    param_grid = {
        "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "Gradient Boosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
        "Naive Bayes": {},
    }

    best_models = {}

    for i, (model_name, model) in enumerate(models.items()):
        grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train.to_numpy().ravel())
        best_models[model_name] = grid_search.best_estimator_

        y_pred = best_models[model_name].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)

        sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", ax=axes[i])
        axes[i].set_xticks([])  # Remove x-axis ticks
        axes[i].set_yticks([])  # Remove y-axis ticks
        axes[i].text(0.5, -0.1, f"{model_name}\nAccuracy: {accuracy:.2f}\nR-squared: {r2:.2f}",
                     ha='center', va='center', transform=axes[i].transAxes, fontsize=10)
    return fig

@app.route('/predictions')
def predictions():
    fig = create_figure_predictions()
    fig.savefig("static/predictions.png")
    return render_template("predictions.html", user_image="static/predictions.png")

if __name__ == "__main__":
    app.run(debug=True)