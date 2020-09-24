{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Copy of Major_Project.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "toc_visible": true,
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/manan-arya/Major_Project/blob/master/main_porject.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oucO9xUK-4TZ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import cross_val_score\n",
        "import random\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.svm import SVR\n",
        "import ga"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4ncydUjIN76X",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "url = \"https://raw.githubusercontent.com/saranshtaneja/genetic_major/master/fb_dataset.csv\"\n",
        "data = pd.read_csv(url)\n",
        "y_train = data.iloc[0:100,701].values\n",
        "x_train = data.iloc[0:100,1:701].values"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JuQ4Q0BD9yrD",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "parameters = {'epsilon':[0,2], 'C':[1, 1000]}"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ib9JA4HJauCE",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "sc_x=StandardScaler()\n",
        "sc_y=StandardScaler()\n",
        "x=sc_x.fit_transform(x_train)\n",
        "y=sc_y.fit_transform(y_train.reshape(-1,1))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LaUynCqLTIWT",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y=y.ravel()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "D6jAz95iIHzc",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "325b5e66-2266-4a86-a192-be1f65524138"
      },
      "source": [
        "svr = SVR(kernel='rbf',C = 10000)\n",
        "clf = GridSearchCV(svr, parameters)\n",
        "clf.fit(x, y)\n",
        "GridSearchCV(estimator=svr,\n",
        "             param_grid={'C': [1, 10000], 'epsilon': [0,2]})\n",
        "score = np.mean(cross_val_score(svr, x, y,  cv=5,  scoring=None))\n",
        "score"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7214387780291002"
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
        "id": "rlKIIz2kZspq",
        "colab_type": "code",
        "outputId": "f040363e-8246-45f0-9c36-2ae424bbfd19",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 806
        }
      },
      "source": [
        "selector = ga.GeneticSelector(estimator=svr, \n",
        "                      n_gen=30, size=100, n_best=20, n_rand=20, \n",
        "                      n_children=5, mutation_rate=0.05)\n",
        "selector.fit(x, y)\n",
        "selector.plot_scores()\n",
        "score = cross_val_score(svr, x[:,selector.support_], y, cv=5, scoring=None)\n",
        "print(\"Score after feature selection: {:.2f}\".format(np.mean(score)))"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0.8231746566382412\n",
            "0.8069173070373065\n",
            "0.7955627441695358\n",
            "0.815273459023447\n",
            "0.817352541424248\n",
            "0.812411402773708\n",
            "0.8185361566771213\n",
            "0.8183269603419141\n",
            "0.8171159020555445\n",
            "0.8183566934362695\n",
            "0.8172240289365285\n",
            "0.8178456735780795\n",
            "0.8333662573401428\n",
            "0.8339869558561638\n",
            "0.8265136759078195\n",
            "0.8261814410126757\n",
            "0.8257889793169824\n",
            "0.8258237064799511\n",
            "0.8303313992588988\n",
            "0.8312246537694982\n",
            "0.8328596417947887\n",
            "0.8320313552776625\n",
            "0.8300498328459242\n",
            "0.8283255993395725\n",
            "0.8294736069512757\n",
            "0.8334625026883581\n",
            "0.8328451870656328\n",
            "0.8321556066525458\n",
            "0.8314920970701651\n",
            "0.831930621474337\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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QgqUFsLBnVWnMcaErNzcXL788ktef/11xo8fz7Rp07jqqqsOH587dy6XXHIJ4EYEHjBgAH379uWqq646PHxKSkoKDz/8MH379uW9997j1VdfpV+/fvTu3Zsrrrji8Dwl69evp3///vTs2ZPHHnuM5s2bH36fyoas95dg9WwPLZ/8yn3Rj3weWrTz7ZzM72HzlzDi9xBWTbxO9JoA79sIrTrUNafGmKpMfcRN3eBP7XrCj56oNsmkSZO48MIL6dGjB61btyYhIYGFCxeSl5dHXFwcEyZM4Nprr2Xv3r386U9/YtasWcTFxfG3v/2NZ5555vCAja1btz487lZWVha33347AI899hivv/4699577+Ge8qNGjeKll146nIeqhqwfMmSIX/4M9kTii+TesOl/8MIA+G6yb+csfhPCIiGtkkr2iqwJsDGN2rhx47j22msBuPbaa3nvvfe48MILmTJlCiUlJXzyySeMHDmSBQsW8N133zFo0CDS0tIYO3YsmzdvPnyda6655vD6ihUrGDx4MD179uTtt99m5cqVAMyfP//w08511113OH1VQ9b7iz2R+OLM2yH1HPjwdnj3Rki7Hi58AmJaVp6++JCrZD/lx9C8hhGHW3VyrbqsU6IxgVXDk0MgZGdn89lnn7F8+XJEhNLSUkSEMWPG8Pzzz5OYmEh6ejotWrRAVTnvvPOqrEeJi4s7vH7zzTczceJEevfuzZtvvsncuXOrzUd1Q9b7gz2R+KptD7jNG3Rx2Th4aRBsnl952pUToSAH0qupZC8XHgmtOtoTiTGN0Pvvv8+NN97I5s2b2bRpE1u3biU1NZWIiAiWLFnCq6++evhppX///vzvf/9j3bp1gJtyd82aNZVe9+DBgyQnJ1NcXMzbb799eH///v354IMPABg/fvzh/VUNWe8vFkiOR3gknPsY3DINJAzevAhm/QFKio5Ol/EGtD4RUgb7dl1rAmxMozRu3Dguv/zyo/ZdccUVjB8/nksuuYSpU6cermhv27Ytb775JqNGjaJXr14MGDCgyubCf/zjHznrrLMYNGgQJ5988uH9zz77LM888wy9evVi3bp1h2dXPP/887nuuusYMGAAPXv25Morr/Tr3CaitWmNFKLS09M1IyPDPxcrPAjTHnVT5yb3hp+8Cm1Pgl0r3NPK+X+Ggff4dq0p97umwg9t8E/ejDEA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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "text": [
            "Score after feature selection: 0.83\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vbrhe2GA6aKt",
        "colab_type": "code",
        "outputId": "79ec9eae-e454-4301-95f7-fcff1d866f26",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        }
      },
      "source": [
        "selector.plot_scores()"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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QgqUFsLBnVWnMcaErNzcXL788ktef/11xo8fz7Rp07jqqqsOH587dy6XXHIJ4EYEHjBgAH379uWqq646PHxKSkoKDz/8MH379uW9997j1VdfpV+/fvTu3Zsrrrji8Dwl69evp3///vTs2ZPHHnuM5s2bH36fyoas95dg9WwPLZ/8yn3Rj3weWrTz7ZzM72HzlzDi9xBWTbxO9JoA79sIrTrUNafGmKpMfcRN3eBP7XrCj56oNsmkSZO48MIL6dGjB61btyYhIYGFCxeSl5dHXFwcEyZM4Nprr2Xv3r386U9/YtasWcTFxfG3v/2NZ5555vCAja1btz487lZWVha33347AI899hivv/4699577+Ge8qNGjeKll146nIeqhqwfMmSIX/4M9kTii+TesOl/8MIA+G6yb+csfhPCIiGtkkr2iqwJsDGN2rhx47j22msBuPbaa3nvvfe48MILmTJlCiUlJXzyySeMHDmSBQsW8N133zFo0CDS0tIYO3YsmzdvPnyda6655vD6ihUrGDx4MD179uTtt99m5cqVAMyfP//w08511113OH1VQ9b7iz2R+OLM2yH1HPjwdnj3Rki7Hi58AmJaVp6++JCrZD/lx9C8hhGHW3VyrbqsU6IxgVXDk0MgZGdn89lnn7F8+XJEhNLSUkSEMWPG8Pzzz5OYmEh6ejotWrRAVTnvvPOqrEeJi4s7vH7zzTczceJEevfuzZtvvsncuXOrzUd1Q9b7gz2R+KptD7jNG3Rx2Th4aRBsnl952pUToSAH0qupZC8XHgmtOtoTiTGN0Pvvv8+NN97I5s2b2bRpE1u3biU1NZWIiAiWLFnCq6++evhppX///vzvf/9j3bp1gJtyd82aNZVe9+DBgyQnJ1NcXMzbb799eH///v354IMPABg/fvzh/VUNWe8vFkiOR3gknPsY3DINJAzevAhm/QFKio5Ol/EGtD4RUgb7dl1rAmxMozRu3Dguv/zyo/ZdccUVjB8/nksuuYSpU6cermhv27Ytb775JqNGjaJXr14MGDCgyubCf/zjHznrrLMYNGgQJ5988uH9zz77LM888wy9evVi3bp1h2dXPP/887nuuusYMGAAPXv25Morr/Tr3CaitWmNFKLS09M1IyPDPxcrPAjTHnVT5yb3hp+8Cm1Pgl0r3NPK+X+Ggff4dq0p97umwg9t8E/ejDEA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\n",
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
        "id": "5qVfdaYlB5Su",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "population = []\n",
        "for i in range(200):\n",
        "  chromosome = np.ones(700, dtype=np.bool)\n",
        "  mask = np.random.rand(len(chromosome)) < 0.3 #The probability 0.3 is chosen arbitrarily, however it is suggested to avoid large probabilities. We would not like to create chromosomes with all variables excluded\n",
        "  chromosome[mask] = False\n",
        "  population.append(chromosome)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eD-6TOTMED9-",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "population = np.array(population)\n",
        "pop=list(population[[0,1,2]])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "B3WdXFtkEGDg",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 330
        },
        "outputId": "a2039c28-d5f9-4f0d-fd06-0952e1ce2f53"
      },
      "source": [
        "sel = ga.GeneticSelector(estimator=svr, \n",
        "                      n_gen=2, size=200, n_best=40, n_rand=40, \n",
        "                      n_children=5, mutation_rate=0.05)\n",
        "sel.fit(x, y.ravel())\n",
        "sel.plot_scores()\n",
        "score = cross_val_score(svr, x[:,sel.support_], y, cv=5, scoring=None)\n",
        "print(\"Score after feature selection: {:.2f}\".format(np.mean(score)))"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0.8243478870716625\n",
            "0.8114690659584006\n"
          ],
          "name": "stdout"
        },
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
        },
        {
          "output_type": "stream",
          "text": [
            "Score after feature selection: 0.81\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZlS6zip8FsmH",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "child =np.array([1,0,1,0,1,1,1,1,0])\n",
        "mask = np.random.rand(len(child))>.5\n",
        "ghp=[1,0,1,0,0,0,0,0,1]\n",
        "for i in range(len(mask)):\n",
        "  if(mask[i]==True):\n",
        "    child[i]=ghp[i]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "R1mHAkD5xnxn",
        "colab_type": "text"
      },
      "source": [
        "##Particle Swarm Optimisation\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YUWmqtNyx-mG",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.model_selection import train_test_split\n",
        "import random"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1YY0Odz8xmvR",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def particle_swarm_optimization(NUMBER_ITERATIONS, NUMBER_PARTICLES, X, y):\n",
        "    # a function to take in a particle, binary list, that will select\n",
        "    # the features coressponding to where the list is a 1 and remove the  \n",
        "    # features that are a zero\n",
        "\n",
        "    def score_particle(particle):\n",
        "        # function to select features from our particle (binary list where 1s are) \n",
        "        # active features and 0's are removed features) then score how\n",
        "        # the selected features are at predicting the dependent variable y\n",
        "        counter = 0\n",
        "        index = []\n",
        "        for a in particle:\n",
        "            if a == 1:\n",
        "                index.append(counter)\n",
        "            counter += 1\n",
        "        active_features = X.iloc[:,index]\n",
        "        X_train,X_test,y_train,y_test = train_test_split(active_features, y)\n",
        "        lr = LogisticRegression()\n",
        "        lr.fit(X_train, y_train)\n",
        "        return lr.score(X_test, y_test)\n",
        "    \n",
        "    def weighted_velocity():\n",
        "        # weight a vector where 30% of the values will be 0's and 70% will be 1s\n",
        "        # this function is so we can add attributes of the center particle to \n",
        "        # non center particles with some random variation, this function adds the random \n",
        "        # variation  for example lets say particle = [0111010], center particle = [1101101],\n",
        "        # the vector returned form this function could be [0011101]\n",
        "        # implying particle + center Particle +weights = [0111010] + [1101101] + [0011101] = \n",
        "        # new particle = [1001000], moving our particle closer to the current center\n",
        "        \n",
        "        weights = [0] * (int(NUMBER_FEATURES * 0.3)) + [1] * (int(NUMBER_FEATURES* 0.7))\n",
        "        velocity = []\n",
        "        for n in np.arange(0 , NUMBER_FEATURES):\n",
        "            velocity.append(random.choice(weights))\n",
        "        return velocity\n",
        "    NUMBER_FEATURES = len(X.columns)\n",
        "    # instantiate where our particles will start (what random features of X they will have)\n",
        "    particles = []\n",
        "    for n in np.arange(1, NUMBER_PARTICLES):\n",
        "        # random select features, each feature has a 50% chance to be activated or removed\n",
        "        particles.append(list(np.random.randint(2,size = NUMBER_FEATURES)))\n",
        "    \n",
        "    #make sure all particles have selected some features\n",
        "    for p in particles:\n",
        "        while sum(p) == 0:\n",
        "            p = list(np.random.randint(2,size = NUMBER_FEATURES))\n",
        "    \n",
        "    # set first center of swarm\n",
        "    current_best_score = score_particle(particles[0])\n",
        "    center_particle = particles[0]\n",
        "    # now ensure the first center before the swarming begins is the\n",
        "    # best center\n",
        "    for p in particles:\n",
        "        # first make sure a feature is selected so we don't test 0 columns\n",
        "        while sum(p) == 0:\n",
        "            p = list(np.random.randint(2,size = NUMBER_FEATURES))\n",
        "        current_test_score = score_particle(p)\n",
        "        if current_test_score > current_best_score:\n",
        "            current_best_score = current_test_score\n",
        "            center_particle = p\n",
        "    # now the center particle is established we can swarm towards it\n",
        "    for iteration in np.arange(1, NUMBER_ITERATIONS):\n",
        "        for p in particles:\n",
        "            #once again make sure we are selecting at least one column\n",
        "            while sum(p) == 0:\n",
        "                p = list(np.random.randint(2,size = NUMBER_FEATURES))\n",
        "            #score partilce\n",
        "            current_test_score = score_particle(p)\n",
        "            if current_test_score > current_best_score:\n",
        "                #update center particle if we have found a new one\n",
        "                current_best_score = current_test_score\n",
        "                center_particle = p\n",
        "                \n",
        "            else:\n",
        "                #otherwise move current particle towards the center with some randomization\n",
        "                p = p + center_particle + weighted_velocity()\n",
        "                \n",
        "    #return center of swarm and resulting prediction score          \n",
        "    return(center_particle, current_best_score)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "n0jRSkUHyWNA",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "Center_Particle , Current_Best_Score = particle_swarm_optimization(10,mask,x_pso,y_pso)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ooCAp9xe0pf2",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "x.shape"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "sju9QKos0qY7",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y.shape"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aVRcnmj700j7",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}