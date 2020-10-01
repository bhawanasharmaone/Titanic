{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Titanic",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "EfiJjSvoObLz"
      },
      "source": [
        "# Suppressing Warnings\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Bz49rDw1O2mY"
      },
      "source": [
        "# Importing Pandas ,NumPy matplotlib\n",
        "import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Bq5fw4i3O2o_",
        "outputId": "4ec966bb-00ac-46cc-f1a9-5a27eee40344",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Importing all datasets\n",
        "url = \"https://raw.githubusercontent.com/bhavna9719/Titanic/master/train.csv\"\n",
        "titanic = pd.read_csv(url)\n",
        "titanic.head()"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked\n",
              "0            1         0       3  ...   7.2500   NaN         S\n",
              "1            2         1       1  ...  71.2833   C85         C\n",
              "2            3         1       3  ...   7.9250   NaN         S\n",
              "3            4         1       1  ...  53.1000  C123         S\n",
              "4            5         0       3  ...   8.0500   NaN         S\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6yqBCJPDQzU8",
        "outputId": "bcd7849f-72d0-49c3-c0f7-6014ee26def7",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Checking rows and columns count\n",
        "titanic.shape"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(891, 12)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vJIjfqJCO2rj",
        "outputId": "780fffaf-55d3-46fc-e6be-0b677e248282",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 354
        }
      },
      "source": [
        "# Checking datatypes and null values of each column\n",
        "titanic.info()"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 891 entries, 0 to 890\n",
            "Data columns (total 12 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  891 non-null    int64  \n",
            " 1   Survived     891 non-null    int64  \n",
            " 2   Pclass       891 non-null    int64  \n",
            " 3   Name         891 non-null    object \n",
            " 4   Sex          891 non-null    object \n",
            " 5   Age          714 non-null    float64\n",
            " 6   SibSp        891 non-null    int64  \n",
            " 7   Parch        891 non-null    int64  \n",
            " 8   Ticket       891 non-null    object \n",
            " 9   Fare         891 non-null    float64\n",
            " 10  Cabin        204 non-null    object \n",
            " 11  Embarked     889 non-null    object \n",
            "dtypes: float64(2), int64(5), object(5)\n",
            "memory usage: 83.7+ KB\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "c9kdK2czO2uP",
        "outputId": "26612869-8296-42a2-bf8f-273cf8708f9a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 287
        }
      },
      "source": [
        "# Checking statistical data of numerical columns\n",
        "titanic.describe()"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>714.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>446.000000</td>\n",
              "      <td>0.383838</td>\n",
              "      <td>2.308642</td>\n",
              "      <td>29.699118</td>\n",
              "      <td>0.523008</td>\n",
              "      <td>0.381594</td>\n",
              "      <td>32.204208</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>257.353842</td>\n",
              "      <td>0.486592</td>\n",
              "      <td>0.836071</td>\n",
              "      <td>14.526497</td>\n",
              "      <td>1.102743</td>\n",
              "      <td>0.806057</td>\n",
              "      <td>49.693429</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.420000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>223.500000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>20.125000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>7.910400</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>446.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>28.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>14.454200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>668.500000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>38.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>31.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>891.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>80.000000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>6.000000</td>\n",
              "      <td>512.329200</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "       PassengerId    Survived      Pclass  ...       SibSp       Parch        Fare\n",
              "count   891.000000  891.000000  891.000000  ...  891.000000  891.000000  891.000000\n",
              "mean    446.000000    0.383838    2.308642  ...    0.523008    0.381594   32.204208\n",
              "std     257.353842    0.486592    0.836071  ...    1.102743    0.806057   49.693429\n",
              "min       1.000000    0.000000    1.000000  ...    0.000000    0.000000    0.000000\n",
              "25%     223.500000    0.000000    2.000000  ...    0.000000    0.000000    7.910400\n",
              "50%     446.000000    0.000000    3.000000  ...    0.000000    0.000000   14.454200\n",
              "75%     668.500000    1.000000    3.000000  ...    1.000000    0.000000   31.000000\n",
              "max     891.000000    1.000000    3.000000  ...    8.000000    6.000000  512.329200\n",
              "\n",
              "[8 rows x 7 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MxqvvWGqO2w7"
      },
      "source": [
        "# Preparing the dataset"
      ],
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eViUIreSO22B"
      },
      "source": [
        "# Drop unnecessary columns\n",
        "titanic.drop([\"PassengerId\", \"Cabin\", \"Name\", \"Ticket\"], axis = 1, inplace = True)"
      ],
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "E29Kz0KkO24m",
        "outputId": "7e557012-3a57-4af1-cdb7-25330dbbbe52",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Checking dataframe after drop\n",
        "titanic.head()"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Survived  Pclass     Sex   Age  SibSp  Parch     Fare Embarked\n",
              "0         0       3    male  22.0      1      0   7.2500        S\n",
              "1         1       1  female  38.0      1      0  71.2833        C\n",
              "2         1       3  female  26.0      0      0   7.9250        S\n",
              "3         1       1  female  35.0      1      0  53.1000        S\n",
              "4         0       3    male  35.0      0      0   8.0500        S"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BlnbM-htO27Z",
        "outputId": "f412d00f-2034-443a-a21b-a632a1064a5b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Filling null values with mean of age\n",
        "age_mean = titanic.Age.mean()\n",
        "titanic.Age.fillna( value = age_mean, inplace = True)\n",
        "titanic.Age.isnull().sum()"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "r2LfAqObO29H",
        "outputId": "61897c39-982d-45b4-bd00-2ea3e7f6216e",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Filling null values with mode \n",
        "titanic.Embarked.fillna( value = titanic.Embarked.mode()[0], inplace = True)\n",
        "titanic.Embarked.isnull().sum()"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hWjC1lLZO3DQ",
        "outputId": "0335966b-7508-4261-c788-4c7612d86ee4",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Re-checking total null values\n",
        "titanic.isnull().sum().sum()"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-beGxxSdO3E_"
      },
      "source": [
        "# Exploratory Data Analysis "
      ],
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "X5HODS-2VRiB",
        "outputId": "005ba51b-4216-4324-cf0b-4b697331ea32",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 496
        }
      },
      "source": [
        "# Ticket class survivals\n",
        "plt.figure( figsize = [10,8])\n",
        "sns.countplot( titanic[\"Pclass\"], hue = titanic[\"Survived\"])\n",
        "plt.show()"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmQAAAHgCAYAAAAL2HHvAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAdT0lEQVR4nO3df7DddX3n8de7SSRWUASubMwNJAragmCsgWrZOlmcCrLdYDtIYLaI1WlohRk60zpap6vUKTu0tXX8tXboYIGWElDqwjKsXURXq1VoYsOvIEMUlWSiBFBqbBGIn/3jfoN3McQbuOd+bnIfj5kz95zP+X7PeZ+Z+8dzvud7zqnWWgAA6Odneg8AADDXCTIAgM4EGQBAZ4IMAKAzQQYA0JkgAwDobH7vAZ6JQw45pC1durT3GAAAP9X69esfaK2N7eq+vTrIli5dmnXr1vUeAwDgp6qqbz7Vfd6yBADoTJABAHQmyAAAOturzyEDAPYNjz32WDZv3pxHHnmk9yjP2MKFCzM+Pp4FCxZMeR9BBgB0t3nz5hxwwAFZunRpqqr3OE9bay0PPvhgNm/enGXLlk15P29ZAgDdPfLIIzn44IP36hhLkqrKwQcfvMdH+gQZADAr7O0xttPTeR2CDACYtS688MIcffTROfbYY7N8+fLcfPPNz/gxr7vuulx00UXTMF2y//77T8vjOIcMAJiVvvSlL+X666/PV77yley333554IEH8uijj05p38cffzzz5+86c1atWpVVq1ZN56jPmCNkAMCstHXr1hxyyCHZb7/9kiSHHHJIXvjCF2bp0qV54IEHkiTr1q3LypUrkyQXXHBBzjrrrJxwwgk566yz8qpXvSp33nnnE4+3cuXKrFu3LpdeemnOO++8PPzwwzn88MPzox/9KEnygx/8IEuWLMljjz2Wr33tazn55JPzyle+Mr/8y7+cr371q0mSe++9N69+9atzzDHH5A//8A+n7bUKMgBgVnrd616X++67Ly95yUvytre9LZ/73Od+6j4bN27Mpz/96Vx55ZVZvXp1rr766iQTcbd169asWLHiiW2f97znZfny5U887vXXX5+TTjopCxYsyJo1a/KhD30o69evz/ve97687W1vS5Kcf/75+Z3f+Z3cfvvtWbRo0bS9VkEGAMxK+++/f9avX5+LL744Y2NjWb16dS699NLd7rNq1ao8+9nPTpKcfvrp+cQnPpEkufrqq3Paaaf9xParV6/OVVddlSRZu3ZtVq9ene3bt+ef/umf8sY3vjHLly/POeeck61btyZJvvjFL+bMM89Mkpx11lnT9VKdQwYAzF7z5s3LypUrs3LlyhxzzDG57LLLMn/+/CfeZnzy10s85znPeeL64sWLc/DBB+e2227LVVddlb/8y7/8icdftWpV3vWud+Whhx7K+vXrc+KJJ+YHP/hBDjzwwGzYsGGXM43i06COkAEAs9Ldd9+de+6554nbGzZsyOGHH56lS5dm/fr1SZJrrrlmt4+xevXq/Omf/mkefvjhHHvssT9x//7775/jjjsu559/fn71V3818+bNy3Of+9wsW7YsH//4x5NMfNnrrbfemiQ54YQTsnbt2iTJFVdcMS2vMxFkAMAstX379px99tk56qijcuyxx2bjxo254IIL8p73vCfnn39+VqxYkXnz5u32MU477bSsXbs2p59++lNus3r16vzt3/5tVq9e/cTaFVdckUsuuSQvf/nLc/TRR+faa69NknzgAx/IRz7ykRxzzDHZsmXL9LzQJNVam7YHm2krVqxo69at6z0GAPAM3XXXXfn5n//53mNMm129nqpa31pbsavtHSEDAOhMkAEAdCbIAAA687UXAOxTXvn2y3uPsFdY/2dv6j0CkzhCBgDQmSADAOhMkAEA7ManPvWpvPSlL80RRxyRiy66aCTP4RwyAGCvMN3nB07lPLodO3bk3HPPzY033pjx8fEcd9xxWbVqVY466qhpncURMgCAp3DLLbfkiCOOyIte9KI861nPyhlnnPHEt/ZPJ0EGAPAUtmzZkiVLljxxe3x8fFp/MmknQQYA0JkgAwB4CosXL8599933xO3Nmzdn8eLF0/48ggwA4Ckcd9xxueeee3Lvvffm0Ucfzdq1a7Nq1appfx6fsgQAeArz58/Phz/84Zx00knZsWNH3vKWt+Too4+e/ueZ9kcEABiBXj/3dMopp+SUU04Z6XN4yxIAoDNBBgDQmSADAOhMkAEAdCbIAAA6E2QAAJ0JMgCAp/CWt7wlL3jBC/Kyl71spM/je8gAgL3Ct957zLQ+3mHvvv2nbvPmN7855513Xt70ptF+B9rIjpBV1cKquqWqbq2qO6vqj4b1S6vq3qraMFyWD+tVVR+s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\n",
            "text/plain": [
              "<Figure size 720x576 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dEbjJwdgVRkc",
        "outputId": "fc84f95c-866b-4d81-f85f-96975bbee62e",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 278
        }
      },
      "source": [
        "# Plotting survival count gender-wise\n",
        "sns.countplot( titanic[\"Sex\"], hue = titanic[\"Survived\"])\n",
        "plt.show()"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYUAAAEGCAYAAACKB4k+AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAUL0lEQVR4nO3df7RV5X3n8fdXQEhEJcJNRrnES6ppIgFJvVqtYxY1bTSMg5kMcjUpwUpKJmqGTqadcWwmEhNnbJo2teoki7VMwIbFD7UTKasxy5hoWzXaew1KBB1JTMKlpAIaomb5C7/zx9k83uJFDnD3PZfL+7XWWez97Ofs8z3LDR/3s/d+TmQmkiQBHNbqAiRJQ4ehIEkqDAVJUmEoSJIKQ0GSVIxsdQEHYsKECdnR0dHqMiTpoNLT07MtM9v623ZQh0JHRwfd3d2tLkOSDioR8dM9bXP4SJJUGAqSpMJQkCQVB/U1BUkaaC+//DK9vb288MILrS7lgI0ZM4b29nZGjRrV9HsMBUnqo7e3lyOPPJKOjg4iotXl7LfMZPv27fT29jJ58uSm3+fwkST18cILLzB+/PiDOhAAIoLx48fv8xmPoSBJuznYA2GX/fkehoIkqTAUJKkJ11xzDVOmTGHatGlMnz6dBx544ID3uXr1aq699toBqA7Gjh07IPs55C80n/LHN7e6hCGj588+1uoSpCHp/vvvZ82aNTz00EOMHj2abdu28dJLLzX13ldeeYWRI/v/p3bWrFnMmjVrIEs9YJ4pSNJebNmyhQkTJjB69GgAJkyYwHHHHUdHRwfbtm0DoLu7mxkzZgCwaNEi5s6dy5lnnsncuXM5/fTTefTRR8v+ZsyYQXd3N0uWLOHyyy9nx44dHH/88bz66qsAPP/880yaNImXX36ZH/3oR5x77rmccsopnHXWWTz22GMAPPnkk5xxxhlMnTqVz3zmMwP2XQ0FSdqLD3zgA2zatIl3vvOdXHrppdxzzz17fc/69ev5zne+w/Lly+nq6mLVqlVAI2C2bNlCZ2dn6Xv00Uczffr0st81a9ZwzjnnMGrUKBYsWMD1119PT08PX/rSl7j00ksBWLhwIZ/85CdZt24dxx577IB9V0NBkvZi7Nix9PT0sHjxYtra2ujq6mLJkiVv+J5Zs2bxpje9CYA5c+Zw6623ArBq1Spmz579uv5dXV2sXLkSgBUrVtDV1cVzzz3HfffdxwUXXMD06dP5xCc+wZYtWwC49957ueiiiwCYO3fuQH1VrylIUjNGjBjBjBkzmDFjBlOnTmXp0qWMHDmyDPns/jzAEUccUZYnTpzI+PHjeeSRR1i5ciVf/epXX7f/WbNmceWVV/L000/T09PD2WefzfPPP8+4ceNYu3ZtvzXVceusZwqStBePP/44TzzxRFlfu3Ytxx9/PB0dHfT09ABw2223veE+urq6+OIXv8iOHTuYNm3a67aPHTuWU089lYULF3LeeecxYsQIjjrqKCZPnswtt9wCNJ5SfvjhhwE488wzWbFiBQDLli0bkO8JhoIk7dVzzz3HvHnzOOmkk5g2bRrr169n0aJFXHXVVSxcuJDOzk5GjBjxhvuYPXs2K1asYM6cOXvs09XVxTe+8Q26urpK27Jly7jppps4+eSTmTJlCrfffjsA1113HTfeeCNTp05l8+bNA/NFgcjMAdvZYOvs7MwD/ZEdb0l9jbekSrBhwwbe/e53t7qMAdPf94mInszs7K+/ZwqSpMJQkCQVhoIkqTAUJEmFoSBJKgwFSVLhE82StI8G+lb2Zm4Hv+OOO1i4cCE7d+7k4x//OFdcccWA1rCLZwqSNMTt3LmTyy67jG9961usX7+e5cuXs379+lo+y1CQpCHuwQcf5IQTTuAd73gHhx9+OBdeeGF5snmgGQqSNMRt3ryZSZMmlfX29vYBndqiL0NBklQYCpI0xE2cOJFNmzaV9d7eXiZOnFjLZxkKkjTEnXrqqTzxxBM8+eSTvPTSS6xYsaK233b2llRJ2keDPaPwyJEjueGGGzjnnHPYuXMnl1xyCVOmTKnns2rZqyRpQM2cOZOZM2fW/jkOH0mSCkNBklQYCpKkwlCQJBW1h0JEjIiIH0TEmmp9ckQ8EBEbI2JlRBxetY+u1jdW2zvqrk2S9K8NxpnCQmBDn/U/Bb6cmScAzwDzq/b5wDNV+5erfpKkQVTrLakR0Q78O+Aa4NMREcDZwEeqLkuBRcBXgPOrZYBbgRsiIjIz66xRkvbVz66eOqD7e/tn1+21zyWXXMKaNWt461vfyg9/+MMB/fy+6j5T+EvgvwGvVuvjgV9k5ivVei+w61nticAmgGr7jqr/vxIRCyKiOyK6t27dWmftkjRkXHzxxdxxxx21f05toRAR5wFPZWbPQO43MxdnZmdmdra1tQ3kriVpyHrf+97HMcccU/vn1Dl8dCYwKyJmAmOAo4DrgHERMbI6G2gHds3/uhmYBPRGxEjgaGB7jfVJknZT25lCZv6PzGzPzA7gQuC7mflR4HvA7KrbPGDXL0Wsrtaptn/X6wmSNLha8ZzCf6dx0XkjjWsGN1XtNwHjq/ZPA/X8AKkkaY8GZUK8zLwbuLta/jFwWj99XgAuGIx6JEn9c5ZUSdpHzdxCOtAuuugi7r77brZt20Z7ezuf+9znmD9//t7fuI8MBUk6CCxfvnxQPse5jyRJhaEgSSoMBUnazXC5G35/voehIEl9jBkzhu3btx/0wZCZbN++nTFjxuzT+7zQLEl9tLe309vby3CYW23MmDG0t7fv03sMBUnqY9SoUUyePLnVZbSMw0eSpMJQkCQVhoIkqTAUJEmFoSBJKgwFSVJhKEiSCkNBklQYCpKkwlCQJBWGgiSpMBQkSYWhIEkqDAVJUmEoSJIKQ0GSVBgKkqTCUJAkFYaCJKkwFCRJhaEgSSoMBUlSYShIkgpDQZJUGAqSpMJQkCQVhoIkqagtFCJiTEQ8GBEPR8SjEfG5qn1yRDwQERsjYmVEHF61j67WN1bbO+qqTZLUvzrPFF4Ezs7Mk4HpwLkRcTrwp8CXM/ME4BlgftV/PvBM1f7lqp8kaRDVFgrZ8Fy1Oqp6JXA2cGvVvhT4ULV8frVOtf39ERF11SdJer1arylExIiIWAs8BdwJ/Aj4RWa+UnXpBSZWyxOBTQDV9h3A+H72uSAiuiOie+vWrXWWL0mHnFpDITN3ZuZ0oB04DXjXAOxzcWZ2ZmZnW1vbAdcoSXrNoNx9lJm/AL4HnAGMi4iR1aZ2YHO1vBmYBFBtPxrYPhj1SZIa6rz7qC0ixlXLbwJ+F9hAIxxmV93mAbdXy6urdart383MrKs+SdLrjdx7l/12LLA0IkbQCJ9VmbkmItYDKyLiC8APgJuq/jcBfx0RG4GngQtrrE2S1I/aQiEzHwHe20/7j2lcX9i9/QXggrrqkSTtnU80S5IKQ0GSVBgKkqTCUJAkFYaCJKkwFCRJhaEgSSqaCoWIuKuZNknSwe0NH16LiDHAm4EJEfEWYNdU1kfx2uymkqRhYm9PNH8C+EPgOKCH10Lhl8ANNdYlSWqBNwyFzLwOuC4iPpWZ1w9STZKkFmlq7qPMvD4ifgvo6PuezLy5prokSS3QVChExF8DvwasBXZWzQkYCpI0jDQ7S2oncJK/byBJw1uzzyn8EPg3dRYiSWq9Zs8UJgDrI+JB4MVdjZk5q5aqJEkt0WwoLKqzCEnS0NDs3Uf31F2IJKn1mr376FkadxsBHA6MAp7PzKPqKkySNPiaPVM4ctdyRARwPnB6XUVJklpjn2dJzYZvAufUUI8kqYWaHT76cJ/Vw2g8t/BCLRVJklqm2buP/n2f5VeAn9AYQpIkDSPNXlP4/boLkSS1XrPDR+3A9cCZVdM/AAszs7euwiRpl59dPbXVJQwZb//sulr33+yF5q8Dq2n8rsJxwN9WbZKkYaTZUGjLzK9n5ivVawnQVmNdkqQWaDYUtkfE70XEiOr1e8D2OguTJA2+ZkPhEmAO8HNgCzAbuLimmiRJLdLsLalXA/My8xmAiDgG+BKNsJAkDRPNnilM2xUIAJn5NPDeekqSJLVKs6FwWES8ZddKdabQ7FmGJOkg0ew/7H8O3B8Rt1TrFwDX1FOSJKlVmn2i+eaI6AbOrpo+nJnr6ytLktQKTQ8BVSFgEEjSMLbPU2c3KyImRcT3ImJ9RDwaEQur9mMi4s6IeKL68y1Ve0TEX0XExoh4JCJ+o67aJEn9qy0UaMym+l8z8yQaP8hzWUScBFwB3JWZJwJ3VesAHwROrF4LgK/UWJskqR+1hUJmbsnMh6rlZ4ENwEQaU24vrbotBT5ULZ8P3Fz9iM/3gXERcWxd9UmSXq/OM4UiIjpoPNfwAPC2zNxSbfo58LZqeSKwqc/bequ23fe1ICK6I6J769attdUsSYei2kMhIsYCtwF/mJm/7LstMxPIfdlfZi7OzM7M7Gxrc04+SRpItYZCRIyiEQjLMvNvquZ/2TUsVP35VNW+GZjU5+3tVZskaZDUefdRADcBGzLzL/psWg3Mq5bnAbf3af9YdRfS6cCOPsNMkqRBUOdUFWcCc4F1EbG2arsSuBZYFRHzgZ/SmH0V4O+AmcBG4FeAPwEqSYOstlDIzH8EYg+b399P/wQuq6seSdLeDcrdR5Kkg4OhIEkqDAVJUmEoSJIKQ0GSVBgKkqTCUJAkFYaCJKkwFCRJhaEgSSoMBUlSYShIkgpDQZJUGAqSpMJQkCQVhoIkqTAUJEmFoSBJKgwFSVJhKEiSCkNBklQYCpKkwlCQJBWGgiSpMBQkSYWhIEkqDAVJUmEoSJIKQ0GSVBgKkqTCUJAkFYaCJKkwFCRJhaEgSSoMBUlSMbKuHUfE14DzgKcy8z1V2zHASqAD+AkwJzOfiYgArgNmAr8CLs7Mh+qqTf372dVTW13CkPH2z65rdQlSS9R5prAEOHe3tiuAuzLzROCuah3gg8CJ1WsB8JUa65Ik7UFtoZCZfw88vVvz+cDSankp8KE+7Tdnw/eBcRFxbF21SZL6N9jXFN6WmVuq5Z8Db6uWJwKb+vTrrdokSYOoZReaMzOB3Nf3RcSCiOiOiO6tW7fWUJkkHboGOxT+ZdewUPXnU1X7ZmBSn37tVdvrZObizOzMzM62trZai5WkQ81gh8JqYF61PA+4vU/7x6LhdGBHn2EmSdIgqfOW1OXADGBCRPQCVwHXAqsiYj7wU2BO1f3vaNyOupHGLam/X1ddkqQ9qy0UMvOiPWx6fz99E7isrlokSc3xiWZJUmEoSJIKQ0GSVNR2TUHSgTnlj29udQlDxv89stUVHDo8U5AkFYaCJKkwFCRJhaEgSSoMBUlSYShIkgpDQZJUGAqSpMJQkCQVhoIkqTAUJEmFoSBJKgwFSVJhKEiSCkNBklQYCpKkwlCQJBWGgiSpMBQkSYWhIEkqDAVJUmEoSJIKQ0GSVBgKkqTCUJAkFYaCJKkwFCRJhaEgSSoMBUlSYShIkgpDQZJUGAqSpGJIhUJEnBsRj0fExoi4otX1SNKhZsiEQkSMAG4EPgicBFwUESe1tipJOrQMmVAATgM2ZuaPM/MlYAVwfotrkqRDyshWF9DHRGBTn/Ve4Dd37xQRC4AF1epzEfH4INR2SDgeJgDbWl3HkHBVtLoC9eGx2cfAHJvH72nDUAqFpmTmYmBxq+sYjiKiOzM7W12HtDuPzcEzlIaPNgOT+qy3V22SpEEylELhn4ATI2JyRBwOXAisbnFNknRIGTLDR5n5SkRcDnwbGAF8LTMfbXFZhxqH5TRUeWwOksjMVtcgSRoihtLwkSSpxQwFSVJhKKhfETEjIta0ug4NDxHxnyNiQ0Qsq2n/iyLij+rY96FmyFxoljSsXQr8Tmb2troQvTHPFIaxiOiIiMciYklE/L+IWBYRvxMR90bEExFxWvW6PyJ+EBH3RcSv97OfIyLiaxHxYNXP6UfUtIj4KvAO4FsR8Sf9HUsRcXFEfDMi7oyIn0TE5RHx6arP9yPimKrfH0TEP0XEwxFxW0S8uZ/P+7WIuCMieiLiHyLiXYP7jQ9uhsLwdwLw58C7qtdHgH8L/BFwJfAYcFZmvhf4LPC/+tnHnwDfzczTgN8G/iwijhiE2jUMZOZ/Av6ZxrFzBHs+lt4DfBg4FbgG+FV1XN4PfKzq8zeZeWpmngxsAOb385GLgU9l5ik0jvP/U883G54cPhr+nszMdQAR8ShwV2ZmRKwDOoCjgaURcSKQwKh+9vEBYFafMdsxwNtp/KWU9sWejiWA72Xms8CzEbED+NuqfR0wrVp+T0R8ARgHjKXxXFMREWOB3wJuiShzBI2u44sMV4bC8Pdin+VX+6y/SuO//+dp/GX8DxHRAdzdzz4C+I+Z6eSDOlD9HksR8Zvs/VgFWAJ8KDMfjoiLgRm77f8w4BeZOX1gyz50OHyko3ltjqmL99Dn28Cnovpfr4h47yDUpeHpQI+lI4EtETEK+OjuGzPzl8CTEXFBtf+IiJMPsOZDiqGgLwL/OyJ+wJ7PHD9PY1jpkWoI6vODVZyGnQM9lv4n8ABwL43rYf35KDA/Ih4GHsXfZdknTnMhSSo8U5AkFYaCJKkwFCRJhaEgSSoMBUlSYShI+6max+fRiHgkItZWD2BJBzWfaJb2Q0ScAZwH/EZmvhgRE4DDW1yWdMA8U5D2z7HAtsx8ESAzt2XmP0fEKRFxTzVD57cj4tiIODoiHt81A21ELI+IP2hp9dIe+PCatB+qidf+EXgz8B1gJXAfcA9wfmZujYgu4JzMvCQifhe4GrgOuDgzz21R6dIbcvhI2g+Z+VxEnAKcRWMK6JXAF2hM/3xnNbXPCGBL1f/Oaj6eGwHn4tGQ5ZmCNAAiYjZwGTAmM8/oZ/thNM4iOoCZu6Yzl4YarylI+yEifr36DYpdptP4fYm26iI0ETEqIqZU2/9Ltf0jwNerWT6lIcczBWk/VENH19P4sZdXgI3AAqAd+CsaU5KPBP4S+Hvgm8BpmflsRPwF8GxmXtWK2qU3YihIkgqHjyRJhaEgSSoMBUlSYShIkgpDQZJUGAqSpMJQkCQV/x8YPikenVtwSQAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "K56idh2_VRnA",
        "outputId": "af187e52-286c-49fb-f3d9-0e51260c399a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 278
        }
      },
      "source": [
        "# Plotting survival count Embarked-wise\n",
        "sns.countplot( titanic[\"Embarked\"], hue = titanic[\"Survived\"])\n",
        "plt.show()"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0AwJzTaIVRqC",
        "outputId": "a036afbe-c61b-4f5e-980d-7c77fb97be9a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 278
        }
      },
      "source": [
        "# Age and survival plot\n",
        "sns.boxplot( titanic[\"Survived\"], titanic[\"Age\"])\n",
        "plt.show()"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uT7aaMa-VRuc",
        "outputId": "8379e6ff-c5ad-4cb0-948f-cacbfc41db11",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 278
        }
      },
      "source": [
        "# Fair and survival plot\n",
        "sns.boxplot( titanic[\"Survived\"], titanic[\"Fare\"])\n",
        "plt.show()"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pue0xjFyVRyW",
        "outputId": "7ce89b3a-738f-427f-f080-9ca6187b99c3",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 486
        }
      },
      "source": [
        "# Checking correlation\n",
        "plt.figure( figsize = [10,8])\n",
        "sns.heatmap(titanic.corr(), annot = True, cmap = \"Greens\")\n",
        "plt.show()"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 720x576 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dHh2t9xWVR1J"
      },
      "source": [
        "# Preparing data for model building"
      ],
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GS-lao2RVR3E",
        "outputId": "7471bd56-8985-446d-ccfd-913b592b2061",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "titanic.head() "
      ],
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Survived  Pclass     Sex   Age  SibSp  Parch     Fare Embarked\n",
              "0         0       3    male  22.0      1      0   7.2500        S\n",
              "1         1       1  female  38.0      1      0  71.2833        C\n",
              "2         1       3  female  26.0      0      0   7.9250        S\n",
              "3         1       1  female  35.0      1      0  53.1000        S\n",
              "4         0       3    male  35.0      0      0   8.0500        S"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "z7-zRgXYctE0",
        "outputId": "e140d724-22d8-4307-b088-b4323348aaac",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Defining the map function\n",
        "variable = [\"Sex\"]\n",
        "\n",
        "def binary_map(x):\n",
        "    return x.map({'female': 1, \"male\": 0})\n",
        "\n",
        "# Applying the function to Sex variable\n",
        "titanic[variable] = titanic[variable].apply(binary_map)\n",
        "titanic.head()"
      ],
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Survived  Pclass  Sex   Age  SibSp  Parch     Fare Embarked\n",
              "0         0       3    0  22.0      1      0   7.2500        S\n",
              "1         1       1    1  38.0      1      0  71.2833        C\n",
              "2         1       3    1  26.0      0      0   7.9250        S\n",
              "3         1       1    1  35.0      1      0  53.1000        S\n",
              "4         0       3    0  35.0      0      0   8.0500        S"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3tBClRN5ctHf",
        "outputId": "9c9d2387-e03a-487c-9927-805445576102",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Defining the map function to define the class lower to higher in numerical terms\n",
        "variable1 = [\"Pclass\"]\n",
        "\n",
        "titanic[variable1] = titanic[variable1].apply(lambda x : x.map({3: 1, 2:2, 1:3}))\n",
        "titanic.head()"
      ],
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Survived  Pclass  Sex   Age  SibSp  Parch     Fare Embarked\n",
              "0         0       1    0  22.0      1      0   7.2500        S\n",
              "1         1       3    1  38.0      1      0  71.2833        C\n",
              "2         1       1    1  26.0      0      0   7.9250        S\n",
              "3         1       3    1  35.0      1      0  53.1000        S\n",
              "4         0       1    0  35.0      0      0   8.0500        S"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QuT3LKnuctKU",
        "outputId": "b5614187-e249-47cf-ef26-78335eaae71a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Creating dummy variables for the variable 'Embarked'. \n",
        "dummy1 = pd.get_dummies(titanic[\"Embarked\"], drop_first=True)\n",
        "\n",
        "# Adding the results to the master dataframe\n",
        "titanic = pd.concat([titanic, dummy1], axis=1)\n",
        "\n",
        "# Dropping the variable of which dummies are created\n",
        "titanic.drop(\"Embarked\", axis = 1, inplace = True)\n",
        "titanic.head()"
      ],
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Q</th>\n",
              "      <th>S</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Survived  Pclass  Sex   Age  SibSp  Parch     Fare  Q  S\n",
              "0         0       1    0  22.0      1      0   7.2500  0  1\n",
              "1         1       3    1  38.0      1      0  71.2833  0  0\n",
              "2         1       1    1  26.0      0      0   7.9250  0  1\n",
              "3         1       3    1  35.0      1      0  53.1000  0  1\n",
              "4         0       1    0  35.0      0      0   8.0500  0  1"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VZY6WN0DctM2"
      },
      "source": [
        "# importing required libraries\n",
        "from sklearn.preprocessing import StandardScaler"
      ],
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "w3YzJ-JXctRX",
        "outputId": "aea2118d-bc1f-4dea-97b3-71c582a5eb35",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Scaling Age and Fare variables\n",
        "scaler = StandardScaler()\n",
        "\n",
        "titanic[['Age','Fare']] = scaler.fit_transform(titanic[['Age','Fare']])\n",
        "\n",
        "titanic.head()"
      ],
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Q</th>\n",
              "      <th>S</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>-0.592481</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>-0.502445</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>0.638789</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0.786845</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>-0.284663</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>-0.488854</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>0.407926</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0.420730</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0.407926</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>-0.486337</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Survived  Pclass  Sex       Age  SibSp  Parch      Fare  Q  S\n",
              "0         0       1    0 -0.592481      1      0 -0.502445  0  1\n",
              "1         1       3    1  0.638789      1      0  0.786845  0  0\n",
              "2         1       1    1 -0.284663      0      0 -0.488854  0  1\n",
              "3         1       3    1  0.407926      1      0  0.420730  0  1\n",
              "4         0       1    0  0.407926      0      0 -0.486337  0  1"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "q_R2V-BOieSZ",
        "outputId": "1cdc50db-3cf3-4c0c-eb0b-ba3d58a7fef6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 287
        }
      },
      "source": [
        "# Checking statistical calc of variables\n",
        "titanic.describe()"
      ],
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Q</th>\n",
              "      <th>S</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>8.910000e+02</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>8.910000e+02</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>0.383838</td>\n",
              "      <td>1.691358</td>\n",
              "      <td>0.352413</td>\n",
              "      <td>2.562796e-16</td>\n",
              "      <td>0.523008</td>\n",
              "      <td>0.381594</td>\n",
              "      <td>-4.373606e-17</td>\n",
              "      <td>0.086420</td>\n",
              "      <td>0.725028</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>0.486592</td>\n",
              "      <td>0.836071</td>\n",
              "      <td>0.477990</td>\n",
              "      <td>1.000562e+00</td>\n",
              "      <td>1.102743</td>\n",
              "      <td>0.806057</td>\n",
              "      <td>1.000562e+00</td>\n",
              "      <td>0.281141</td>\n",
              "      <td>0.446751</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>-2.253155e+00</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>-6.484217e-01</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>-5.924806e-01</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>-4.891482e-01</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000e+00</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>-3.573909e-01</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>4.079260e-01</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>-2.424635e-02</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.870872e+00</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>6.000000</td>\n",
              "      <td>9.667167e+00</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "         Survived      Pclass         Sex  ...          Fare           Q           S\n",
              "count  891.000000  891.000000  891.000000  ...  8.910000e+02  891.000000  891.000000\n",
              "mean     0.383838    1.691358    0.352413  ... -4.373606e-17    0.086420    0.725028\n",
              "std      0.486592    0.836071    0.477990  ...  1.000562e+00    0.281141    0.446751\n",
              "min      0.000000    1.000000    0.000000  ... -6.484217e-01    0.000000    0.000000\n",
              "25%      0.000000    1.000000    0.000000  ... -4.891482e-01    0.000000    0.000000\n",
              "50%      0.000000    1.000000    0.000000  ... -3.573909e-01    0.000000    1.000000\n",
              "75%      1.000000    2.000000    1.000000  ... -2.424635e-02    0.000000    1.000000\n",
              "max      1.000000    3.000000    1.000000  ...  9.667167e+00    1.000000    1.000000\n",
              "\n",
              "[8 rows x 9 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rcKpy3aXctUR",
        "outputId": "1f3f0d38-2101-4445-c670-b61df4fb960d",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 486
        }
      },
      "source": [
        "# Re-checking correlations with survival\n",
        "plt.figure( figsize = [10,8])\n",
        "sns.heatmap(titanic.corr(), annot = True, cmap = \"Greens\")\n",
        "plt.show()"
      ],
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 720x576 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wEWjdoz_ctYZ",
        "outputId": "ed86efcf-16b8-4b2d-b0bc-ac09d70b9d4f",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 70
        }
      },
      "source": [
        "# Checking suruvival rate\n",
        "titanic.Survived.value_counts( normalize = True) * 100"
      ],
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    61.616162\n",
              "1    38.383838\n",
              "Name: Survived, dtype: float64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SPm7x8LyctcA"
      },
      "source": [
        "# Preparing dataframe for model building\n",
        "\n",
        "y_train = titanic.pop('Survived')\n",
        "X_train = titanic"
      ],
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Rh6-46NEctec",
        "outputId": "c2d06068-e330-4036-8d4c-18b50ab7c3b9",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 230
        }
      },
      "source": [
        "# Checking heads\n",
        "print(X_train.head())\n",
        "print(y_train.head())"
      ],
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "   Pclass  Sex       Age  SibSp  Parch      Fare  Q  S\n",
            "0       1    0 -0.592481      1      0 -0.502445  0  1\n",
            "1       3    1  0.638789      1      0  0.786845  0  0\n",
            "2       1    1 -0.284663      0      0 -0.488854  0  1\n",
            "3       3    1  0.407926      1      0  0.420730  0  1\n",
            "4       1    0  0.407926      0      0 -0.486337  0  1\n",
            "0    0\n",
            "1    1\n",
            "2    1\n",
            "3    1\n",
            "4    0\n",
            "Name: Survived, dtype: int64\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "J8Bks0W9kRR_"
      },
      "source": [
        "# Importing required libraries\n",
        "import statsmodels.api as sm"
      ],
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "po_9MNOKctXY",
        "outputId": "d694b522-055d-4a71-c930-295b208941aa",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 373
        }
      },
      "source": [
        "# Building model 1\n",
        "X_train_sm = sm.add_constant(X_train)\n",
        "logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())\n",
        "res = logm1.fit()\n",
        "res.summary()"
      ],
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<table class=\"simpletable\">\n",
              "<caption>Generalized Linear Model Regression Results</caption>\n",
              "<tr>\n",
              "  <th>Dep. Variable:</th>       <td>Survived</td>     <th>  No. Observations:  </th>  <td>   891</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   882</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>     8</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Link Function:</th>         <td>logit</td>      <th>  Scale:             </th> <td>  1.0000</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -392.39</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Date:</th>            <td>Thu, 01 Oct 2020</td> <th>  Deviance:          </th> <td>  784.78</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Time:</th>                <td>12:04:18</td>     <th>  Pearson chi2:      </th>  <td>  904.</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>No. Iterations:</th>          <td>5</td>        <th>                     </th>     <td> </td>   \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
              "</tr>\n",
              "</table>\n",
              "<table class=\"simpletable\">\n",
              "<tr>\n",
              "     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>const</th>  <td>   -2.9642</td> <td>    0.365</td> <td>   -8.120</td> <td> 0.000</td> <td>   -3.680</td> <td>   -2.249</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Pclass</th> <td>    1.1022</td> <td>    0.144</td> <td>    7.675</td> <td> 0.000</td> <td>    0.821</td> <td>    1.384</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Sex</th>    <td>    2.7272</td> <td>    0.201</td> <td>   13.597</td> <td> 0.000</td> <td>    2.334</td> <td>    3.120</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Age</th>    <td>   -0.5154</td> <td>    0.102</td> <td>   -5.061</td> <td> 0.000</td> <td>   -0.715</td> <td>   -0.316</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>SibSp</th>  <td>   -0.3269</td> <td>    0.110</td> <td>   -2.985</td> <td> 0.003</td> <td>   -0.542</td> <td>   -0.112</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Parch</th>  <td>   -0.0946</td> <td>    0.119</td> <td>   -0.797</td> <td> 0.426</td> <td>   -0.327</td> <td>    0.138</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Fare</th>   <td>    0.0974</td> <td>    0.118</td> <td>    0.823</td> <td> 0.410</td> <td>   -0.134</td> <td>    0.329</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Q</th>      <td>   -0.0327</td> <td>    0.382</td> <td>   -0.085</td> <td> 0.932</td> <td>   -0.782</td> <td>    0.717</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>S</th>      <td>   -0.4139</td> <td>    0.237</td> <td>   -1.748</td> <td> 0.081</td> <td>   -0.878</td> <td>    0.050</td>\n",
              "</tr>\n",
              "</table>"
            ],
            "text/plain": [
              "<class 'statsmodels.iolib.summary.Summary'>\n",
              "\"\"\"\n",
              "                 Generalized Linear Model Regression Results                  \n",
              "==============================================================================\n",
              "Dep. Variable:               Survived   No. Observations:                  891\n",
              "Model:                            GLM   Df Residuals:                      882\n",
              "Model Family:                Binomial   Df Model:                            8\n",
              "Link Function:                  logit   Scale:                          1.0000\n",
              "Method:                          IRLS   Log-Likelihood:                -392.39\n",
              "Date:                Thu, 01 Oct 2020   Deviance:                       784.78\n",
              "Time:                        12:04:18   Pearson chi2:                     904.\n",
              "No. Iterations:                     5                                         \n",
              "Covariance Type:            nonrobust                                         \n",
              "==============================================================================\n",
              "                 coef    std err          z      P>|z|      [0.025      0.975]\n",
              "------------------------------------------------------------------------------\n",
              "const         -2.9642      0.365     -8.120      0.000      -3.680      -2.249\n",
              "Pclass         1.1022      0.144      7.675      0.000       0.821       1.384\n",
              "Sex            2.7272      0.201     13.597      0.000       2.334       3.120\n",
              "Age           -0.5154      0.102     -5.061      0.000      -0.715      -0.316\n",
              "SibSp         -0.3269      0.110     -2.985      0.003      -0.542      -0.112\n",
              "Parch         -0.0946      0.119     -0.797      0.426      -0.327       0.138\n",
              "Fare           0.0974      0.118      0.823      0.410      -0.134       0.329\n",
              "Q             -0.0327      0.382     -0.085      0.932      -0.782       0.717\n",
              "S             -0.4139      0.237     -1.748      0.081      -0.878       0.050\n",
              "==============================================================================\n",
              "\"\"\""
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XpASZgWZctP_"
      },
      "source": [
        "# Dropping high p-value variable\n",
        "X_train.drop(\"Q\", axis = 1, inplace = True)"
      ],
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZhhznxZ3VRs1",
        "outputId": "982ebc55-56fc-42bc-987d-2835726a138d",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 355
        }
      },
      "source": [
        "# Building model 2\n",
        "X_train_sm = sm.add_constant(X_train)\n",
        "logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())\n",
        "res = logm2.fit()\n",
        "res.summary()"
      ],
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<table class=\"simpletable\">\n",
              "<caption>Generalized Linear Model Regression Results</caption>\n",
              "<tr>\n",
              "  <th>Dep. Variable:</th>       <td>Survived</td>     <th>  No. Observations:  </th>  <td>   891</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   883</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>     7</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Link Function:</th>         <td>logit</td>      <th>  Scale:             </th> <td>  1.0000</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -392.39</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Date:</th>            <td>Thu, 01 Oct 2020</td> <th>  Deviance:          </th> <td>  784.79</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Time:</th>                <td>12:04:18</td>     <th>  Pearson chi2:      </th>  <td>  905.</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>No. Iterations:</th>          <td>5</td>        <th>                     </th>     <td> </td>   \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
              "</tr>\n",
              "</table>\n",
              "<table class=\"simpletable\">\n",
              "<tr>\n",
              "     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>const</th>  <td>   -2.9792</td> <td>    0.321</td> <td>   -9.293</td> <td> 0.000</td> <td>   -3.608</td> <td>   -2.351</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Pclass</th> <td>    1.1049</td> <td>    0.140</td> <td>    7.894</td> <td> 0.000</td> <td>    0.831</td> <td>    1.379</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Sex</th>    <td>    2.7253</td> <td>    0.199</td> <td>   13.676</td> <td> 0.000</td> <td>    2.335</td> <td>    3.116</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Age</th>    <td>   -0.5160</td> <td>    0.102</td> <td>   -5.081</td> <td> 0.000</td> <td>   -0.715</td> <td>   -0.317</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>SibSp</th>  <td>   -0.3269</td> <td>    0.109</td> <td>   -2.986</td> <td> 0.003</td> <td>   -0.542</td> <td>   -0.112</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Parch</th>  <td>   -0.0937</td> <td>    0.118</td> <td>   -0.792</td> <td> 0.428</td> <td>   -0.325</td> <td>    0.138</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Fare</th>   <td>    0.0980</td> <td>    0.118</td> <td>    0.829</td> <td> 0.407</td> <td>   -0.134</td> <td>    0.329</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>S</th>      <td>   -0.4034</td> <td>    0.203</td> <td>   -1.990</td> <td> 0.047</td> <td>   -0.801</td> <td>   -0.006</td>\n",
              "</tr>\n",
              "</table>"
            ],
            "text/plain": [
              "<class 'statsmodels.iolib.summary.Summary'>\n",
              "\"\"\"\n",
              "                 Generalized Linear Model Regression Results                  \n",
              "==============================================================================\n",
              "Dep. Variable:               Survived   No. Observations:                  891\n",
              "Model:                            GLM   Df Residuals:                      883\n",
              "Model Family:                Binomial   Df Model:                            7\n",
              "Link Function:                  logit   Scale:                          1.0000\n",
              "Method:                          IRLS   Log-Likelihood:                -392.39\n",
              "Date:                Thu, 01 Oct 2020   Deviance:                       784.79\n",
              "Time:                        12:04:18   Pearson chi2:                     905.\n",
              "No. Iterations:                     5                                         \n",
              "Covariance Type:            nonrobust                                         \n",
              "==============================================================================\n",
              "                 coef    std err          z      P>|z|      [0.025      0.975]\n",
              "------------------------------------------------------------------------------\n",
              "const         -2.9792      0.321     -9.293      0.000      -3.608      -2.351\n",
              "Pclass         1.1049      0.140      7.894      0.000       0.831       1.379\n",
              "Sex            2.7253      0.199     13.676      0.000       2.335       3.116\n",
              "Age           -0.5160      0.102     -5.081      0.000      -0.715      -0.317\n",
              "SibSp         -0.3269      0.109     -2.986      0.003      -0.542      -0.112\n",
              "Parch         -0.0937      0.118     -0.792      0.428      -0.325       0.138\n",
              "Fare           0.0980      0.118      0.829      0.407      -0.134       0.329\n",
              "S             -0.4034      0.203     -1.990      0.047      -0.801      -0.006\n",
              "==============================================================================\n",
              "\"\"\""
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Vq_c-0ovkv3E"
      },
      "source": [
        "# Dropping high p-value variable\n",
        "X_train.drop(\"Parch\", axis = 1, inplace = True)"
      ],
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jSqQrMDxkv7v",
        "outputId": "d8d04ffd-b690-48ea-cc42-47a5e27563c3",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 337
        }
      },
      "source": [
        "# Building model 3\n",
        "X_train_sm = sm.add_constant(X_train)\n",
        "logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())\n",
        "res = logm3.fit()\n",
        "res.summary()"
      ],
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<table class=\"simpletable\">\n",
              "<caption>Generalized Linear Model Regression Results</caption>\n",
              "<tr>\n",
              "  <th>Dep. Variable:</th>       <td>Survived</td>     <th>  No. Observations:  </th>  <td>   891</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   884</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>     6</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Link Function:</th>         <td>logit</td>      <th>  Scale:             </th> <td>  1.0000</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -392.71</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Date:</th>            <td>Thu, 01 Oct 2020</td> <th>  Deviance:          </th> <td>  785.42</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Time:</th>                <td>12:04:18</td>     <th>  Pearson chi2:      </th>  <td>  910.</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>No. Iterations:</th>          <td>5</td>        <th>                     </th>     <td> </td>   \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
              "</tr>\n",
              "</table>\n",
              "<table class=\"simpletable\">\n",
              "<tr>\n",
              "     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>const</th>  <td>   -3.0063</td> <td>    0.318</td> <td>   -9.443</td> <td> 0.000</td> <td>   -3.630</td> <td>   -2.382</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Pclass</th> <td>    1.1165</td> <td>    0.139</td> <td>    8.038</td> <td> 0.000</td> <td>    0.844</td> <td>    1.389</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Sex</th>    <td>    2.6936</td> <td>    0.195</td> <td>   13.841</td> <td> 0.000</td> <td>    2.312</td> <td>    3.075</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Age</th>    <td>   -0.5123</td> <td>    0.101</td> <td>   -5.059</td> <td> 0.000</td> <td>   -0.711</td> <td>   -0.314</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>SibSp</th>  <td>   -0.3493</td> <td>    0.106</td> <td>   -3.287</td> <td> 0.001</td> <td>   -0.558</td> <td>   -0.141</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Fare</th>   <td>    0.0774</td> <td>    0.113</td> <td>    0.685</td> <td> 0.493</td> <td>   -0.144</td> <td>    0.299</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>S</th>      <td>   -0.4160</td> <td>    0.202</td> <td>   -2.061</td> <td> 0.039</td> <td>   -0.812</td> <td>   -0.020</td>\n",
              "</tr>\n",
              "</table>"
            ],
            "text/plain": [
              "<class 'statsmodels.iolib.summary.Summary'>\n",
              "\"\"\"\n",
              "                 Generalized Linear Model Regression Results                  \n",
              "==============================================================================\n",
              "Dep. Variable:               Survived   No. Observations:                  891\n",
              "Model:                            GLM   Df Residuals:                      884\n",
              "Model Family:                Binomial   Df Model:                            6\n",
              "Link Function:                  logit   Scale:                          1.0000\n",
              "Method:                          IRLS   Log-Likelihood:                -392.71\n",
              "Date:                Thu, 01 Oct 2020   Deviance:                       785.42\n",
              "Time:                        12:04:18   Pearson chi2:                     910.\n",
              "No. Iterations:                     5                                         \n",
              "Covariance Type:            nonrobust                                         \n",
              "==============================================================================\n",
              "                 coef    std err          z      P>|z|      [0.025      0.975]\n",
              "------------------------------------------------------------------------------\n",
              "const         -3.0063      0.318     -9.443      0.000      -3.630      -2.382\n",
              "Pclass         1.1165      0.139      8.038      0.000       0.844       1.389\n",
              "Sex            2.6936      0.195     13.841      0.000       2.312       3.075\n",
              "Age           -0.5123      0.101     -5.059      0.000      -0.711      -0.314\n",
              "SibSp         -0.3493      0.106     -3.287      0.001      -0.558      -0.141\n",
              "Fare           0.0774      0.113      0.685      0.493      -0.144       0.299\n",
              "S             -0.4160      0.202     -2.061      0.039      -0.812      -0.020\n",
              "==============================================================================\n",
              "\"\"\""
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 37
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "w0tzBMNLkwRn"
      },
      "source": [
        "# Dropping high p-value variable\n",
        "X_train.drop(\"Fare\", axis = 1, inplace = True)"
      ],
      "execution_count": 38,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "r-n2bxmTkv6Q",
        "outputId": "16a95f3c-914b-47c4-e22f-bf52a479ee09",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 320
        }
      },
      "source": [
        "# Building model 4\n",
        "X_train_sm = sm.add_constant(X_train)\n",
        "logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())\n",
        "res = logm4.fit()\n",
        "res.summary()"
      ],
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<table class=\"simpletable\">\n",
              "<caption>Generalized Linear Model Regression Results</caption>\n",
              "<tr>\n",
              "  <th>Dep. Variable:</th>       <td>Survived</td>     <th>  No. Observations:  </th>  <td>   891</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   885</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>     5</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Link Function:</th>         <td>logit</td>      <th>  Scale:             </th> <td>  1.0000</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -392.96</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Date:</th>            <td>Thu, 01 Oct 2020</td> <th>  Deviance:          </th> <td>  785.91</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Time:</th>                <td>12:04:18</td>     <th>  Pearson chi2:      </th>  <td>  912.</td> \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>No. Iterations:</th>          <td>5</td>        <th>                     </th>     <td> </td>   \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
              "</tr>\n",
              "</table>\n",
              "<table class=\"simpletable\">\n",
              "<tr>\n",
              "     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
              "</tr>\n",
              "<tr>\n",
              "  <th>const</th>  <td>   -3.0860</td> <td>    0.297</td> <td>  -10.406</td> <td> 0.000</td> <td>   -3.667</td> <td>   -2.505</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Pclass</th> <td>    1.1651</td> <td>    0.120</td> <td>    9.693</td> <td> 0.000</td> <td>    0.930</td> <td>    1.401</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Sex</th>    <td>    2.7007</td> <td>    0.194</td> <td>   13.901</td> <td> 0.000</td> <td>    2.320</td> <td>    3.082</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>Age</th>    <td>   -0.5159</td> <td>    0.101</td> <td>   -5.104</td> <td> 0.000</td> <td>   -0.714</td> <td>   -0.318</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>SibSp</th>  <td>   -0.3343</td> <td>    0.104</td> <td>   -3.222</td> <td> 0.001</td> <td>   -0.538</td> <td>   -0.131</td>\n",
              "</tr>\n",
              "<tr>\n",
              "  <th>S</th>      <td>   -0.4414</td> <td>    0.198</td> <td>   -2.225</td> <td> 0.026</td> <td>   -0.830</td> <td>   -0.053</td>\n",
              "</tr>\n",
              "</table>"
            ],
            "text/plain": [
              "<class 'statsmodels.iolib.summary.Summary'>\n",
              "\"\"\"\n",
              "                 Generalized Linear Model Regression Results                  \n",
              "==============================================================================\n",
              "Dep. Variable:               Survived   No. Observations:                  891\n",
              "Model:                            GLM   Df Residuals:                      885\n",
              "Model Family:                Binomial   Df Model:                            5\n",
              "Link Function:                  logit   Scale:                          1.0000\n",
              "Method:                          IRLS   Log-Likelihood:                -392.96\n",
              "Date:                Thu, 01 Oct 2020   Deviance:                       785.91\n",
              "Time:                        12:04:18   Pearson chi2:                     912.\n",
              "No. Iterations:                     5                                         \n",
              "Covariance Type:            nonrobust                                         \n",
              "==============================================================================\n",
              "                 coef    std err          z      P>|z|      [0.025      0.975]\n",
              "------------------------------------------------------------------------------\n",
              "const         -3.0860      0.297    -10.406      0.000      -3.667      -2.505\n",
              "Pclass         1.1651      0.120      9.693      0.000       0.930       1.401\n",
              "Sex            2.7007      0.194     13.901      0.000       2.320       3.082\n",
              "Age           -0.5159      0.101     -5.104      0.000      -0.714      -0.318\n",
              "SibSp         -0.3343      0.104     -3.222      0.001      -0.538      -0.131\n",
              "S             -0.4414      0.198     -2.225      0.026      -0.830      -0.053\n",
              "==============================================================================\n",
              "\"\"\""
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8UNSSniglZ2M"
      },
      "source": [
        "# Importing required library\n",
        "from statsmodels.stats.outliers_influence import variance_inflation_factor"
      ],
      "execution_count": 40,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WIP1okkmlj3O",
        "outputId": "2654ed45-eb0b-4d7e-8db7-980761ab8f34",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Create a dataframe that will contain the names of all the feature variables and their respective VIFs\n",
        "\n",
        "vif = pd.DataFrame()\n",
        "vif['Features'] = X_train.columns\n",
        "vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]\n",
        "vif['VIF'] = round(vif['VIF'], 2)\n",
        "vif = vif.sort_values(by = \"VIF\", ascending = False)\n",
        "vif"
      ],
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Features</th>\n",
              "      <th>VIF</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Pclass</td>\n",
              "      <td>2.96</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>S</td>\n",
              "      <td>2.41</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Sex</td>\n",
              "      <td>1.59</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>SibSp</td>\n",
              "      <td>1.30</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Age</td>\n",
              "      <td>1.15</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "  Features   VIF\n",
              "0   Pclass  2.96\n",
              "4        S  2.41\n",
              "1      Sex  1.59\n",
              "3    SibSp  1.30\n",
              "2      Age  1.15"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wl8fnP8ilj52",
        "outputId": "3907047c-f3f8-4051-cc84-b6271e195a2d",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 52
        }
      },
      "source": [
        "# Checking predicted values\n",
        "y_train_pred = res.predict(X_train_sm).values.reshape(-1)\n",
        "y_train_pred[:10]"
      ],
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0.08385815, 0.92027575, 0.61897882, 0.89319244, 0.07091022,\n",
              "       0.12775841, 0.26955491, 0.09399972, 0.60957279, 0.90324265])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 42
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cEQc1xvllj9A",
        "outputId": "1b4d40c2-4155-42f2-f67e-6b9829aa354c",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# new dataframe to compare actual and predicted\n",
        "y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Survival_Prob':y_train_pred})\n",
        "y_train_pred_final.head()"
      ],
      "execution_count": 43,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Survival_Prob</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0.083858</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0.920276</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>0.618979</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>0.893192</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>0.070910</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Survived  Survival_Prob\n",
              "0         0       0.083858\n",
              "1         1       0.920276\n",
              "2         1       0.618979\n",
              "3         1       0.893192\n",
              "4         0       0.070910"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 43
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GB_cbdsKlkB3",
        "outputId": "875cd47d-1459-4d15-b4cc-4494edf40623",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Adding column \"Predicted\"\n",
        "y_train_pred_final['predicted'] = y_train_pred_final.Survival_Prob.map(lambda x: 1 if x > 0.5 else 0)\n",
        "\n",
        "# Let's see the head\n",
        "y_train_pred_final.head()"
      ],
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Survival_Prob</th>\n",
              "      <th>predicted</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0.083858</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0.920276</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>0.618979</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>0.893192</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>0.070910</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Survived  Survival_Prob  predicted\n",
              "0         0       0.083858          0\n",
              "1         1       0.920276          1\n",
              "2         1       0.618979          1\n",
              "3         1       0.893192          1\n",
              "4         0       0.070910          0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 44
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eK3sT8UvlkGb",
        "outputId": "5634244f-1610-44df-f4e6-50ef9430e2a4",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 52
        }
      },
      "source": [
        "# Importing required library\n",
        "from sklearn import metrics\n",
        "\n",
        "# Confusion matrix \n",
        "confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )\n",
        "print(confusion)"
      ],
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[471  78]\n",
            " [ 98 244]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VyHf-MEKlkLi"
      },
      "source": [
        "# Predicted     not_sur    sur\n",
        "# Actual\n",
        "# not_sur        471      78\n",
        "# sur            98       244  "
      ],
      "execution_count": 46,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hLY4P4LzlkOa",
        "outputId": "d2bee03e-d472-42cd-9caf-19d049e628f1",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Let's check the overall accuracy.\n",
        "print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted))"
      ],
      "execution_count": 47,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0.8024691358024691\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Zw325OiplkJa"
      },
      "source": [
        "TP = confusion[1,1] # true positive \n",
        "TN = confusion[0,0] # true negatives\n",
        "FP = confusion[0,1] # false positives\n",
        "FN = confusion[1,0] # false negatives"
      ],
      "execution_count": 48,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pAHCaWHAlkEh",
        "outputId": "47bed8cf-427e-48a5-f81a-9de89389fe5c",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Let's see the sensitivity of our logistic regression model \n",
        "TP / float(TP+FN)"
      ],
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7134502923976608"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 49
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "53HoRT9TlkAE",
        "outputId": "2c5c6cc1-0a4f-4d48-d72e-0b84d0af4610",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Let us calculate specificity\n",
        "TN / float(TN+FP)"
      ],
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8579234972677595"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 50
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DknvNgShsSSw",
        "outputId": "830fbd76-37bc-4303-9a5c-c42d56c53202",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Calculate false postive rate - predicting churn when customer does not have churned\n",
        "FP/ float(TN+FP)"
      ],
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.14207650273224043"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 51
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Neoy4V0ssSem",
        "outputId": "af2291fc-d941-44f3-b717-acf2ed511471",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# positive predictive value \n",
        "TP / float(TP+FP)"
      ],
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7577639751552795"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qYPfFKtpsSjY",
        "outputId": "df9e2ca5-0b01-40c1-ac3b-35009134cccd",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Negative predictive value\n",
        "TN / float(TN+ FN)"
      ],
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.827768014059754"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 53
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0tBs_oTFstEF"
      },
      "source": [
        "# Defining draw_roc function \n",
        "def draw_roc( actual, probs ):\n",
        "    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,\n",
        "                                              drop_intermediate = False )\n",
        "    auc_score = metrics.roc_auc_score( actual, probs )\n",
        "    plt.figure(figsize=(5, 5))\n",
        "    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )\n",
        "    plt.plot([0, 1], [0, 1], 'k--')\n",
        "    plt.xlim([0.0, 1.0])\n",
        "    plt.ylim([0.0, 1.05])\n",
        "    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')\n",
        "    plt.ylabel('True Positive Rate')\n",
        "    plt.title('Receiver operating characteristic example')\n",
        "    plt.legend(loc=\"lower right\")\n",
        "    plt.show()\n",
        "\n",
        "    return None"
      ],
      "execution_count": 54,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7QlKMMMrstG0"
      },
      "source": [
        "# Assigning values\n",
        "fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Survived, y_train_pred_final.Survival_Prob, drop_intermediate = False )"
      ],
      "execution_count": 55,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ymXgy_rvstLo",
        "outputId": "8866b0a5-48e4-45d8-f817-26fdbaa226e9",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 350
        }
      },
      "source": [
        "# Calling the draw_roc func\n",
        "draw_roc(y_train_pred_final.Survived, y_train_pred_final.Survival_Prob)"
      ],
      "execution_count": 56,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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dGOMMbt27TL33XefadCggalbt64ZM2aMSU1NTZt38eLF5vbbbzcNGzY0jRo1Mq+++upN64+OjjZPPPGEadKkiWnevLn57rvvjDHGfPvtt6ZOnTqmTZs2Zvjw4ebJJ580xhjz5JNPmm+//faGdWzevNkAZurUqWmvRUREmAEDBphmzZqZRo0apbV8p9e0aVNz9epVY4wxGzduNPXr1zctW7Y0b775pqlZs6Yxxpivv/7aDB8+PG2Z2NhYM2zYMNO0aVPTuHFj88ADDxhjjDl48KBp1qyZad68uRk9evRN792tiIuLM/369TN169Y1d955pzly5IgxxpjTp0+b7t27p833zjvvmNtuu800adLEDB482MTHxxtjjImMjDQ9evQwTZs2NW3btr3hCoUHHnjA7Nq166Zt5nWrvRina+I8gX+V+mb+8nU80LxKhtMPHz7M6NGj+frr6qALGQAADTtJREFUrylVqlQ+R6dyav/+/WklC+UeEyZMICgoiGeeecbuUPLV+fPnefTRR1m1atVN0zI67kRkqzGm9a1sq1BV7Xfu3EmHDh1Yt25d2iUhShV1zz//PP7+/naHke9OnjzJ+PHj82VbhabVfsOGDTzwwAMEBQWxZs0aLeUo5RAQEMDjjz9udxj57s4778y3bRWKEunq1asJCQmhUqVKbNiwQZOoh/G000vKs7njeCsUibRevXqEhIQQFhZGcHCw3eGoHAgICODSpUuaTFW+MMbqjzQgICBP1+vRVfsVK1Zw//33ExwczMKFC+0OR92C6tWrEx4eTkREhN2hqCLieg/5ecmtiVREugEfA97AFGPMP9JN9wemA3cAl4CBxpjj2a3XGMO4ceN48803mTRpEs8991zeB6/yha+vb572VK6UHdyWSEXEG/gMCAHCgc0issgYs89ptqeBSGNMPRF5BPgQGJjlig18Of49vp/xBYMHD2bo0KFu2gOllHKNO8+R3gUcNsYcNcYkArOB3unm6Q1MczyeB9wv2dzDmXz1PN/P+IIXXniBadOmaYfMSinbuTORVgNOOT0Pd7yW4TzGmGQgCihHFlLjYhg5+g0+/vjjtG6zlFLKTh7R2CQiw4BhjqcJ//7nuD3//uc4O0Nyp/LARbuDcKPCvH+Fed+g8O/fbbe6oDsT6WmghtPz6o7XMponXER8gFJYjU43MMZMBiYDiMiWW72N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\n",
            "text/plain": [
              "<Figure size 360x360 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8UK_I7s3stRj",
        "outputId": "96313a12-6a77-4dac-c7de-7557403e1c0d",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Let's create columns with different probability cutoffs \n",
        "numbers = [float(x)/10 for x in range(10)]\n",
        "for i in numbers:\n",
        "    y_train_pred_final[i]= y_train_pred_final.Survival_Prob.map(lambda x: 1 if x > i else 0)\n",
        "y_train_pred_final.head()"
      ],
      "execution_count": 57,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Survival_Prob</th>\n",
              "      <th>predicted</th>\n",
              "      <th>0.0</th>\n",
              "      <th>0.1</th>\n",
              "      <th>0.2</th>\n",
              "      <th>0.3</th>\n",
              "      <th>0.4</th>\n",
              "      <th>0.5</th>\n",
              "      <th>0.6</th>\n",
              "      <th>0.7</th>\n",
              "      <th>0.8</th>\n",
              "      <th>0.9</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0.083858</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0.920276</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>0.618979</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>0.893192</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>0.070910</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Survived  Survival_Prob  predicted  0.0  0.1  ...  0.5  0.6  0.7  0.8  0.9\n",
              "0         0       0.083858          0    1    0  ...    0    0    0    0    0\n",
              "1         1       0.920276          1    1    1  ...    1    1    1    1    1\n",
              "2         1       0.618979          1    1    1  ...    1    1    0    0    0\n",
              "3         1       0.893192          1    1    1  ...    1    1    1    1    0\n",
              "4         0       0.070910          0    1    0  ...    0    0    0    0    0\n",
              "\n",
              "[5 rows x 13 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 57
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bUkbP6cgstdV",
        "outputId": "800ce787-093e-4440-8fe7-a0b02457c4f7",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 212
        }
      },
      "source": [
        "# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.\n",
        "cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])\n",
        "from sklearn.metrics import confusion_matrix\n",
        "\n",
        "# TP = confusion[1,1] # true positive \n",
        "# TN = confusion[0,0] # true negatives\n",
        "# FP = confusion[0,1] # false positives\n",
        "# FN = confusion[1,0] # false negatives\n",
        "\n",
        "num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]\n",
        "for i in num:\n",
        "    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )\n",
        "    total1=sum(sum(cm1))\n",
        "    accuracy = (cm1[0,0]+cm1[1,1])/total1\n",
        "    \n",
        "    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]\n",
        "print(cutoff_df)"
      ],
      "execution_count": 58,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "     prob  accuracy     sensi     speci\n",
            "0.0   0.0  0.383838  1.000000  0.000000\n",
            "0.1   0.1  0.557800  0.923977  0.329690\n",
            "0.2   0.2  0.712682  0.850877  0.626594\n",
            "0.3   0.3  0.771044  0.824561  0.737705\n",
            "0.4   0.4  0.789001  0.763158  0.805100\n",
            "0.5   0.5  0.802469  0.713450  0.857923\n",
            "0.6   0.6  0.814815  0.657895  0.912568\n",
            "0.7   0.7  0.792368  0.505848  0.970856\n",
            "0.8   0.8  0.756453  0.380117  0.990893\n",
            "0.9   0.9  0.691358  0.204678  0.994536\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Do6kqceNstbF",
        "outputId": "d7620afb-8646-4a89-81c7-f03cc9cbbd5e",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 278
        }
      },
      "source": [
        "# Let's plot accuracy sensitivity and specificity for various probabilities.\n",
        "cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])\n",
        "plt.show()"
      ],
      "execution_count": 59,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "y6C-xvqYstXQ",
        "outputId": "d8d3c254-814c-454d-ad4f-2ded7bbd45bc",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Adding column final predicted\n",
        "y_train_pred_final['final_predicted'] = y_train_pred_final.Survival_Prob.map( lambda x: 1 if x > 0.33 else 0)\n",
        "\n",
        "y_train_pred_final.head()"
      ],
      "execution_count": 60,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Survival_Prob</th>\n",
              "      <th>predicted</th>\n",
              "      <th>0.0</th>\n",
              "      <th>0.1</th>\n",
              "      <th>0.2</th>\n",
              "      <th>0.3</th>\n",
              "      <th>0.4</th>\n",
              "      <th>0.5</th>\n",
              "      <th>0.6</th>\n",
              "      <th>0.7</th>\n",
              "      <th>0.8</th>\n",
              "      <th>0.9</th>\n",
              "      <th>final_predicted</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0.083858</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0.920276</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>0.618979</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>0.893192</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>0.070910</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Survived  Survival_Prob  predicted  0.0  ...  0.7  0.8  0.9  final_predicted\n",
              "0         0       0.083858          0    1  ...    0    0    0                0\n",
              "1         1       0.920276          1    1  ...    1    1    1                1\n",
              "2         1       0.618979          1    1  ...    0    0    0                1\n",
              "3         1       0.893192          1    1  ...    1    1    0                1\n",
              "4         0       0.070910          0    1  ...    0    0    0                0\n",
              "\n",
              "[5 rows x 14 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 60
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qre_PVixstVp",
        "outputId": "6a60a939-6720-4b4c-8a44-af18f7260e73",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Let's check the overall accuracy.\n",
        "metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)"
      ],
      "execution_count": 61,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7833894500561167"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 61
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZqCuHtdDstP8",
        "outputId": "b4b864f7-20ce-4110-e905-4990c511710b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 52
        }
      },
      "source": [
        "confusion2 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.final_predicted )\n",
        "confusion2"
      ],
      "execution_count": 62,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[422, 127],\n",
              "       [ 66, 276]])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 62
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bMdQwQmgutYJ"
      },
      "source": [
        "TP = confusion2[1,1] # true positive \n",
        "TN = confusion2[0,0] # true negatives\n",
        "FP = confusion2[0,1] # false positives\n",
        "FN = confusion2[1,0] # false negatives"
      ],
      "execution_count": 63,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vF-nF9RWutdY",
        "outputId": "5fd94c78-d5cc-429d-c25b-b6e83f9e6983",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Let's see the sensitivity of our logistic regression model\n",
        "TP / float(TP+FN)"
      ],
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8070175438596491"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 64
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eKjYWGODutjC",
        "outputId": "eb199d79-217e-44fd-a5bf-b508928a7da1",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Let us calculate specificity\n",
        "TN / float(TN+FP)"
      ],
      "execution_count": 65,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7686703096539163"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 65
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1_DEtOQHuthm",
        "outputId": "567cec2a-aeec-4ccd-dca9-5c72fb58d1c8",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Calculate false postive rate\n",
        "FP/ float(TN+FP)"
      ],
      "execution_count": 66,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.23132969034608378"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 66
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-MKJsoxRutbX",
        "outputId": "62be5beb-6890-4333-ba60-6fdbfcc2af21",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Positive predictive value \n",
        "TP / float(TP+FP)"
      ],
      "execution_count": 67,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.684863523573201"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 67
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fwG4fIuautVc",
        "outputId": "cc2916a7-83ca-41e9-e90d-2b9cbcb23b42",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Negative predictive value\n",
        "TN / float(TN+ FN)"
      ],
      "execution_count": 68,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8647540983606558"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 68
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oNisfNY5stKK"
      },
      "source": [
        "# Recall and Precision"
      ],
      "execution_count": 69,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2s2MbhOpsShk"
      },
      "source": [
        "# importing required libraries\n",
        "from sklearn.metrics import precision_recall_curve"
      ],
      "execution_count": 70,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cGaDZckKvOnQ",
        "outputId": "b5a37b28-1220-434f-b97c-e16c31b6f78d",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 212
        }
      },
      "source": [
        "# Printing head of required columns\n",
        "y_train_pred_final.Survived.head(), y_train_pred_final.predicted.head()"
      ],
      "execution_count": 71,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0    0\n",
              " 1    1\n",
              " 2    1\n",
              " 3    1\n",
              " 4    0\n",
              " Name: Survived, dtype: int64, 0    0\n",
              " 1    1\n",
              " 2    1\n",
              " 3    1\n",
              " 4    0\n",
              " Name: predicted, dtype: int64)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 71
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qitmbLCJvOs4"
      },
      "source": [
        "# Assigning values\n",
        "p, r, thresholds = precision_recall_curve(y_train_pred_final.Survived, y_train_pred_final.Survival_Prob)"
      ],
      "execution_count": 72,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "OTsMU3MPvOrB",
        "outputId": "238ff32b-91c8-4eaa-ee65-016486e3953f",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 264
        }
      },
      "source": [
        "# Plotting recall and precision\n",
        "plt.plot(thresholds, p[:-1], \"g-\")\n",
        "plt.plot(thresholds, r[:-1], \"r-\")\n",
        "plt.show()"
      ],
      "execution_count": 73,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "abkGrBiAw-HO"
      },
      "source": [
        "Making predictions on test dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mfzuBU7OvOkk"
      },
      "source": [
        "# importing test file\n",
        "url1 = \"https://raw.githubusercontent.com/bhavna9719/Titanic/master/test.csv\"\n",
        "titanic_test = pd.read_csv(url1)"
      ],
      "execution_count": 74,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mbbkpAoQ2jtQ",
        "outputId": "d4a53f4a-2024-4f1c-d61e-8db3c87ff8a3",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# New dataframe for submission\n",
        "Submission = pd.DataFrame(({'PassengerId':titanic_test.PassengerId}))\n",
        "Submission.head()"
      ],
      "execution_count": 75,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>892</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>893</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>894</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>895</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>896</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId\n",
              "0          892\n",
              "1          893\n",
              "2          894\n",
              "3          895\n",
              "4          896"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 75
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1Z9aCYvSvOiM",
        "outputId": "11ee7bd5-6cc9-4e7f-93e0-66cc307614f1",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Checking the dataset\n",
        "titanic_test.head()"
      ],
      "execution_count": 76,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>892</td>\n",
              "      <td>3</td>\n",
              "      <td>Kelly, Mr. James</td>\n",
              "      <td>male</td>\n",
              "      <td>34.5</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>330911</td>\n",
              "      <td>7.8292</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Q</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>893</td>\n",
              "      <td>3</td>\n",
              "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
              "      <td>female</td>\n",
              "      <td>47.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>363272</td>\n",
              "      <td>7.0000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>894</td>\n",
              "      <td>2</td>\n",
              "      <td>Myles, Mr. Thomas Francis</td>\n",
              "      <td>male</td>\n",
              "      <td>62.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>240276</td>\n",
              "      <td>9.6875</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Q</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>895</td>\n",
              "      <td>3</td>\n",
              "      <td>Wirz, Mr. Albert</td>\n",
              "      <td>male</td>\n",
              "      <td>27.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>315154</td>\n",
              "      <td>8.6625</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>896</td>\n",
              "      <td>3</td>\n",
              "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
              "      <td>female</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3101298</td>\n",
              "      <td>12.2875</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Pclass  ... Cabin Embarked\n",
              "0          892       3  ...   NaN        Q\n",
              "1          893       3  ...   NaN        S\n",
              "2          894       2  ...   NaN        Q\n",
              "3          895       3  ...   NaN        S\n",
              "4          896       3  ...   NaN        S\n",
              "\n",
              "[5 rows x 11 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 76
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Nx6LxbZ2zIqx",
        "outputId": "e349e9b8-f880-4736-fbbe-e238f6e01eb9",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Rows and columns count\n",
        "titanic_test.shape"
      ],
      "execution_count": 77,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(418, 11)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 77
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kz-d70yzxsFd",
        "outputId": "2465448f-1e25-4293-fdfe-46d56302bbb2",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 336
        }
      },
      "source": [
        "# Checking null values and datatypes\n",
        "titanic_test.info()"
      ],
      "execution_count": 78,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 418 entries, 0 to 417\n",
            "Data columns (total 11 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  418 non-null    int64  \n",
            " 1   Pclass       418 non-null    int64  \n",
            " 2   Name         418 non-null    object \n",
            " 3   Sex          418 non-null    object \n",
            " 4   Age          332 non-null    float64\n",
            " 5   SibSp        418 non-null    int64  \n",
            " 6   Parch        418 non-null    int64  \n",
            " 7   Ticket       418 non-null    object \n",
            " 8   Fare         417 non-null    float64\n",
            " 9   Cabin        91 non-null     object \n",
            " 10  Embarked     418 non-null    object \n",
            "dtypes: float64(2), int64(4), object(5)\n",
            "memory usage: 36.0+ KB\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ct5JCzq3yHip",
        "outputId": "89d5aee7-2b91-4ca2-b429-b02472d6a671",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Filling null values with mean of age\n",
        "titanic_test.Age.fillna( value = age_mean, inplace = True)\n",
        "titanic_test.Age.isnull().sum()"
      ],
      "execution_count": 79,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 79
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vD_6Cgic1jtu",
        "outputId": "d98e6831-eff6-44a5-9eff-42e561a715ae",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Applying the function to Sex variable\n",
        "titanic_test[variable] = titanic_test[variable].apply(binary_map)\n",
        "titanic_test.head()"
      ],
      "execution_count": 80,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>892</td>\n",
              "      <td>3</td>\n",
              "      <td>Kelly, Mr. James</td>\n",
              "      <td>0</td>\n",
              "      <td>34.5</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>330911</td>\n",
              "      <td>7.8292</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Q</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>893</td>\n",
              "      <td>3</td>\n",
              "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
              "      <td>1</td>\n",
              "      <td>47.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>363272</td>\n",
              "      <td>7.0000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>894</td>\n",
              "      <td>2</td>\n",
              "      <td>Myles, Mr. Thomas Francis</td>\n",
              "      <td>0</td>\n",
              "      <td>62.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>240276</td>\n",
              "      <td>9.6875</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Q</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>895</td>\n",
              "      <td>3</td>\n",
              "      <td>Wirz, Mr. Albert</td>\n",
              "      <td>0</td>\n",
              "      <td>27.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>315154</td>\n",
              "      <td>8.6625</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>896</td>\n",
              "      <td>3</td>\n",
              "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
              "      <td>1</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3101298</td>\n",
              "      <td>12.2875</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Pclass  ... Cabin  Embarked\n",
              "0          892       3  ...   NaN         Q\n",
              "1          893       3  ...   NaN         S\n",
              "2          894       2  ...   NaN         Q\n",
              "3          895       3  ...   NaN         S\n",
              "4          896       3  ...   NaN         S\n",
              "\n",
              "[5 rows x 11 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 80
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xO3K7xXI1jpI",
        "outputId": "17504f54-6340-413e-baab-92b22c8f571c",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Defining the map function to define the class lower to higher in numerical term\n",
        "titanic_test[variable1] = titanic_test[variable1].apply(lambda x : x.map({3: 1, 2:2, 1:3}))\n",
        "titanic_test.head()"
      ],
      "execution_count": 81,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>892</td>\n",
              "      <td>1</td>\n",
              "      <td>Kelly, Mr. James</td>\n",
              "      <td>0</td>\n",
              "      <td>34.5</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>330911</td>\n",
              "      <td>7.8292</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Q</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>893</td>\n",
              "      <td>1</td>\n",
              "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
              "      <td>1</td>\n",
              "      <td>47.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>363272</td>\n",
              "      <td>7.0000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>894</td>\n",
              "      <td>2</td>\n",
              "      <td>Myles, Mr. Thomas Francis</td>\n",
              "      <td>0</td>\n",
              "      <td>62.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>240276</td>\n",
              "      <td>9.6875</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Q</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>895</td>\n",
              "      <td>1</td>\n",
              "      <td>Wirz, Mr. Albert</td>\n",
              "      <td>0</td>\n",
              "      <td>27.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>315154</td>\n",
              "      <td>8.6625</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>896</td>\n",
              "      <td>1</td>\n",
              "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
              "      <td>1</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3101298</td>\n",
              "      <td>12.2875</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Pclass  ... Cabin  Embarked\n",
              "0          892       1  ...   NaN         Q\n",
              "1          893       1  ...   NaN         S\n",
              "2          894       2  ...   NaN         Q\n",
              "3          895       1  ...   NaN         S\n",
              "4          896       1  ...   NaN         S\n",
              "\n",
              "[5 rows x 11 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 81
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2DeAPYwC3Vxi",
        "outputId": "c327d959-271f-499a-f08a-8fe3be94729b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Creating dummy variables for the variable 'Embarked'. \n",
        "dummy1 = pd.get_dummies(titanic_test[\"Embarked\"], drop_first=True)\n",
        "\n",
        "# Adding the results to the master dataframe\n",
        "titanic_test = pd.concat([titanic_test, dummy1], axis=1)\n",
        "\n",
        "# Dropping the variable of which dummies are created\n",
        "titanic_test.drop(\"Embarked\", axis = 1, inplace = True)\n",
        "titanic_test.head()"
      ],
      "execution_count": 82,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Q</th>\n",
              "      <th>S</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>892</td>\n",
              "      <td>1</td>\n",
              "      <td>Kelly, Mr. James</td>\n",
              "      <td>0</td>\n",
              "      <td>34.5</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>330911</td>\n",
              "      <td>7.8292</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>893</td>\n",
              "      <td>1</td>\n",
              "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
              "      <td>1</td>\n",
              "      <td>47.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>363272</td>\n",
              "      <td>7.0000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>894</td>\n",
              "      <td>2</td>\n",
              "      <td>Myles, Mr. Thomas Francis</td>\n",
              "      <td>0</td>\n",
              "      <td>62.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>240276</td>\n",
              "      <td>9.6875</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>895</td>\n",
              "      <td>1</td>\n",
              "      <td>Wirz, Mr. Albert</td>\n",
              "      <td>0</td>\n",
              "      <td>27.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>315154</td>\n",
              "      <td>8.6625</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>896</td>\n",
              "      <td>1</td>\n",
              "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
              "      <td>1</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3101298</td>\n",
              "      <td>12.2875</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Pclass  ...  Q  S\n",
              "0          892       1  ...  1  0\n",
              "1          893       1  ...  0  1\n",
              "2          894       2  ...  1  0\n",
              "3          895       1  ...  0  1\n",
              "4          896       1  ...  0  1\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 82
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tg16B3iY3VnM",
        "outputId": "1e8e3d65-0951-41ae-a6d7-1d5b086cc41f",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "titanic_test[['Age','Fare']] = scaler.transform(titanic_test[['Age','Fare']])\n",
        "titanic_test.head()"
      ],
      "execution_count": 83,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Q</th>\n",
              "      <th>S</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>892</td>\n",
              "      <td>1</td>\n",
              "      <td>Kelly, Mr. James</td>\n",
              "      <td>0</td>\n",
              "      <td>0.369449</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>330911</td>\n",
              "      <td>-0.490783</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>893</td>\n",
              "      <td>1</td>\n",
              "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
              "      <td>1</td>\n",
              "      <td>1.331378</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>363272</td>\n",
              "      <td>-0.507479</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>894</td>\n",
              "      <td>2</td>\n",
              "      <td>Myles, Mr. Thomas Francis</td>\n",
              "      <td>0</td>\n",
              "      <td>2.485693</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>240276</td>\n",
              "      <td>-0.453367</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>895</td>\n",
              "      <td>1</td>\n",
              "      <td>Wirz, Mr. Albert</td>\n",
              "      <td>0</td>\n",
              "      <td>-0.207709</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>315154</td>\n",
              "      <td>-0.474005</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>896</td>\n",
              "      <td>1</td>\n",
              "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
              "      <td>1</td>\n",
              "      <td>-0.592481</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3101298</td>\n",
              "      <td>-0.401017</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Pclass  ...  Q  S\n",
              "0          892       1  ...  1  0\n",
              "1          893       1  ...  0  1\n",
              "2          894       2  ...  1  0\n",
              "3          895       1  ...  0  1\n",
              "4          896       1  ...  0  1\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 83
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cf3g9INb3Vj7"
      },
      "source": [
        "# Drop unnecessary columns\n",
        "titanic_test.drop([\"PassengerId\", \"Cabin\", \"Name\", \"Ticket\", \"Q\", \"Parch\", \"Fare\", ], axis = 1, inplace = True)"
      ],
      "execution_count": 84,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6fwOoNFj3Vf9",
        "outputId": "655e3890-5937-45a6-cb05-22d937b5373e",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 287
        }
      },
      "source": [
        "# Checking scaled variables\n",
        "titanic_test.describe()"
      ],
      "execution_count": 85,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>S</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>418.000000</td>\n",
              "      <td>418.000000</td>\n",
              "      <td>418.000000</td>\n",
              "      <td>418.000000</td>\n",
              "      <td>418.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>1.734450</td>\n",
              "      <td>0.363636</td>\n",
              "      <td>0.035052</td>\n",
              "      <td>0.447368</td>\n",
              "      <td>0.645933</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>0.841838</td>\n",
              "      <td>0.481622</td>\n",
              "      <td>0.972446</td>\n",
              "      <td>0.896760</td>\n",
              "      <td>0.478803</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>-2.272394</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>-0.515526</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>3.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.465642</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>3.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.563054</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "           Pclass         Sex         Age       SibSp           S\n",
              "count  418.000000  418.000000  418.000000  418.000000  418.000000\n",
              "mean     1.734450    0.363636    0.035052    0.447368    0.645933\n",
              "std      0.841838    0.481622    0.972446    0.896760    0.478803\n",
              "min      1.000000    0.000000   -2.272394    0.000000    0.000000\n",
              "25%      1.000000    0.000000   -0.515526    0.000000    0.000000\n",
              "50%      1.000000    0.000000    0.000000    0.000000    1.000000\n",
              "75%      3.000000    1.000000    0.465642    1.000000    1.000000\n",
              "max      3.000000    1.000000    3.563054    8.000000    1.000000"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 85
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "q-453fA13Vcv"
      },
      "source": [
        "# Adding constant\n",
        "X_test_sm = sm.add_constant(titanic_test)"
      ],
      "execution_count": 86,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uE8Z8C3L3VZp",
        "outputId": "b3387234-6278-42a5-cf4d-4e7a2c74c0c4",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 212
        }
      },
      "source": [
        "# predicting and checking predicted values\n",
        "y_test_pred = res.predict(X_test_sm)\n",
        "y_test_pred[:10]"
      ],
      "execution_count": 87,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    0.107982\n",
              "1    0.335634\n",
              "2    0.115255\n",
              "3    0.094902\n",
              "4    0.576804\n",
              "5    0.149427\n",
              "6    0.683054\n",
              "7    0.200248\n",
              "8    0.776300\n",
              "9    0.063823\n",
              "dtype: float64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 87
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "J2fgoyrb0e_n",
        "outputId": "8d392a77-5e76-400b-b45d-862023fa9e10",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        }
      },
      "source": [
        "# Adding new column of final prediction\n",
        "Submission['Survived'] = y_test_pred.map(lambda x: 1 if x > 0.33 else 0)\n",
        "Submission.head()"
      ],
      "execution_count": 88,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>892</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>893</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>894</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>895</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>896</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Survived\n",
              "0          892         0\n",
              "1          893         1\n",
              "2          894         0\n",
              "3          895         0\n",
              "4          896         1"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 88
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "R1Sx19Wx7JCU",
        "outputId": "07f079c0-b6cc-423b-d6cb-9facffd6a76b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 70
        }
      },
      "source": [
        "# Value counts of 0 and 1\n",
        "Submission.Survived.value_counts( normalize = True)"
      ],
      "execution_count": 89,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    0.528708\n",
              "1    0.471292\n",
              "Name: Survived, dtype: float64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 89
        }
      ]
    }
  ]
}