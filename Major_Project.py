{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Major_Project.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/manan-arya/Major_Project/blob/pso%2B/Major_Project.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
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
        "import matplotlib.pyplot as plt"
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
        "url = \"https://raw.githubusercontent.com/manan-arya/Major_Project/pso%2B/after_pso.csv\""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uwF-yud__Sls",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "data = pd.read_csv(url)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ic9tTAcQ_elA",
        "colab_type": "code",
        "outputId": "ef63e79b-9c25-4fcd-ed84-da9a58fbaa99",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 426
        }
      },
      "source": [
        "data"
      ],
      "execution_count": 35,
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
              "      <th>dominant_color.histogram[4]</th>\n",
              "      <th>dominant_color.histogram[9]</th>\n",
              "      <th>general.fps</th>\n",
              "      <th>other.clutter_metric</th>\n",
              "      <th>title.sentiment</th>\n",
              "      <th>views.1</th>\n",
              "      <th>comments.2</th>\n",
              "      <th>comments.3</th>\n",
              "      <th>likes.3</th>\n",
              "      <th>likes.10</th>\n",
              "      <th>shares.11</th>\n",
              "      <th>comments.11</th>\n",
              "      <th>shares.12</th>\n",
              "      <th>comments.12</th>\n",
              "      <th>shares.13</th>\n",
              "      <th>likes.14</th>\n",
              "      <th>shares.15</th>\n",
              "      <th>comments.15</th>\n",
              "      <th>views.16</th>\n",
              "      <th>views.17</th>\n",
              "      <th>shares.17</th>\n",
              "      <th>comments.17</th>\n",
              "      <th>likes.17</th>\n",
              "      <th>shares.18</th>\n",
              "      <th>views.19</th>\n",
              "      <th>likes.19</th>\n",
              "      <th>views.20</th>\n",
              "      <th>likes.20</th>\n",
              "      <th>views.21</th>\n",
              "      <th>comments.21</th>\n",
              "      <th>views.22</th>\n",
              "      <th>likes.22</th>\n",
              "      <th>views.23</th>\n",
              "      <th>shares.23</th>\n",
              "      <th>likes.23</th>\n",
              "      <th>views.24</th>\n",
              "      <th>comments.24</th>\n",
              "      <th>likes.24</th>\n",
              "      <th>views.25</th>\n",
              "      <th>shares.25</th>\n",
              "      <th>...</th>\n",
              "      <th>likes.150</th>\n",
              "      <th>views.151</th>\n",
              "      <th>shares.151</th>\n",
              "      <th>likes.151</th>\n",
              "      <th>shares.152</th>\n",
              "      <th>comments.152</th>\n",
              "      <th>likes.152</th>\n",
              "      <th>likes.153</th>\n",
              "      <th>comments.154</th>\n",
              "      <th>likes.154</th>\n",
              "      <th>views.155</th>\n",
              "      <th>shares.156</th>\n",
              "      <th>comments.156</th>\n",
              "      <th>likes.156</th>\n",
              "      <th>shares.157</th>\n",
              "      <th>comments.157</th>\n",
              "      <th>likes.157</th>\n",
              "      <th>shares.158</th>\n",
              "      <th>likes.158</th>\n",
              "      <th>likes.159</th>\n",
              "      <th>shares.160</th>\n",
              "      <th>likes.160</th>\n",
              "      <th>views.161</th>\n",
              "      <th>likes.161</th>\n",
              "      <th>likes.162</th>\n",
              "      <th>comments.163</th>\n",
              "      <th>likes.163</th>\n",
              "      <th>views.164</th>\n",
              "      <th>comments.164</th>\n",
              "      <th>likes.164</th>\n",
              "      <th>shares.165</th>\n",
              "      <th>comments.165</th>\n",
              "      <th>likes.165</th>\n",
              "      <th>views.166</th>\n",
              "      <th>comments.166</th>\n",
              "      <th>views.167</th>\n",
              "      <th>shares.168</th>\n",
              "      <th>comments.168</th>\n",
              "      <th>likes.168</th>\n",
              "      <th>views.168</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.098788</td>\n",
              "      <td>25</td>\n",
              "      <td>0.161410</td>\n",
              "      <td>4.651</td>\n",
              "      <td>21057</td>\n",
              "      <td>105</td>\n",
              "      <td>130</td>\n",
              "      <td>1256</td>\n",
              "      <td>2239</td>\n",
              "      <td>1690</td>\n",
              "      <td>219</td>\n",
              "      <td>1758</td>\n",
              "      <td>224</td>\n",
              "      <td>1834</td>\n",
              "      <td>2511</td>\n",
              "      <td>2018</td>\n",
              "      <td>235</td>\n",
              "      <td>128464</td>\n",
              "      <td>138659</td>\n",
              "      <td>2190</td>\n",
              "      <td>248</td>\n",
              "      <td>2706</td>\n",
              "      <td>2269</td>\n",
              "      <td>148963</td>\n",
              "      <td>2852</td>\n",
              "      <td>152664</td>\n",
              "      <td>2930</td>\n",
              "      <td>160000</td>\n",
              "      <td>265</td>\n",
              "      <td>168228</td>\n",
              "      <td>3106</td>\n",
              "      <td>177844</td>\n",
              "      <td>2833</td>\n",
              "      <td>3196</td>\n",
              "      <td>187894</td>\n",
              "      <td>286</td>\n",
              "      <td>3265</td>\n",
              "      <td>196543</td>\n",
              "      <td>3073</td>\n",
              "      <td>...</td>\n",
              "      <td>4220</td>\n",
              "      <td>349837</td>\n",
              "      <td>4911</td>\n",
              "      <td>4220</td>\n",
              "      <td>4912</td>\n",
              "      <td>375</td>\n",
              "      <td>4220</td>\n",
              "      <td>4220</td>\n",
              "      <td>376</td>\n",
              "      <td>4220</td>\n",
              "      <td>350162</td>\n",
              "      <td>4913</td>\n",
              "      <td>376</td>\n",
              "      <td>4223</td>\n",
              "      <td>4913</td>\n",
              "      <td>376</td>\n",
              "      <td>4223</td>\n",
              "      <td>4913</td>\n",
              "      <td>4223</td>\n",
              "      <td>4223</td>\n",
              "      <td>4914</td>\n",
              "      <td>4223</td>\n",
              "      <td>350577</td>\n",
              "      <td>4223</td>\n",
              "      <td>4223</td>\n",
              "      <td>376</td>\n",
              "      <td>4223</td>\n",
              "      <td>350785</td>\n",
              "      <td>376</td>\n",
              "      <td>4223</td>\n",
              "      <td>4916</td>\n",
              "      <td>376</td>\n",
              "      <td>4224</td>\n",
              "      <td>350942</td>\n",
              "      <td>376</td>\n",
              "      <td>351045</td>\n",
              "      <td>4916</td>\n",
              "      <td>376</td>\n",
              "      <td>4225</td>\n",
              "      <td>351137</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.027848</td>\n",
              "      <td>25</td>\n",
              "      <td>0.106316</td>\n",
              "      <td>4.566</td>\n",
              "      <td>15618</td>\n",
              "      <td>33</td>\n",
              "      <td>39</td>\n",
              "      <td>822</td>\n",
              "      <td>1162</td>\n",
              "      <td>402</td>\n",
              "      <td>49</td>\n",
              "      <td>406</td>\n",
              "      <td>49</td>\n",
              "      <td>413</td>\n",
              "      <td>1269</td>\n",
              "      <td>425</td>\n",
              "      <td>50</td>\n",
              "      <td>55830</td>\n",
              "      <td>56640</td>\n",
              "      <td>437</td>\n",
              "      <td>50</td>\n",
              "      <td>1328</td>\n",
              "      <td>443</td>\n",
              "      <td>60093</td>\n",
              "      <td>1368</td>\n",
              "      <td>61800</td>\n",
              "      <td>1386</td>\n",
              "      <td>63632</td>\n",
              "      <td>53</td>\n",
              "      <td>65154</td>\n",
              "      <td>1416</td>\n",
              "      <td>66526</td>\n",
              "      <td>494</td>\n",
              "      <td>1426</td>\n",
              "      <td>67896</td>\n",
              "      <td>55</td>\n",
              "      <td>1440</td>\n",
              "      <td>69044</td>\n",
              "      <td>508</td>\n",
              "      <td>...</td>\n",
              "      <td>1617</td>\n",
              "      <td>87823</td>\n",
              "      <td>578</td>\n",
              "      <td>1617</td>\n",
              "      <td>578</td>\n",
              "      <td>61</td>\n",
              "      <td>1617</td>\n",
              "      <td>1617</td>\n",
              "      <td>61</td>\n",
              "      <td>1617</td>\n",
              "      <td>87891</td>\n",
              "      <td>578</td>\n",
              "      <td>61</td>\n",
              "      <td>1617</td>\n",
              "      <td>578</td>\n",
              "      <td>61</td>\n",
              "      <td>1617</td>\n",
              "      <td>578</td>\n",
              "      <td>1617</td>\n",
              "      <td>1617</td>\n",
              "      <td>578</td>\n",
              "      <td>1617</td>\n",
              "      <td>88020</td>\n",
              "      <td>1617</td>\n",
              "      <td>1617</td>\n",
              "      <td>61</td>\n",
              "      <td>1617</td>\n",
              "      <td>88066</td>\n",
              "      <td>61</td>\n",
              "      <td>1617</td>\n",
              "      <td>578</td>\n",
              "      <td>61</td>\n",
              "      <td>1617</td>\n",
              "      <td>88110</td>\n",
              "      <td>61</td>\n",
              "      <td>88128</td>\n",
              "      <td>579</td>\n",
              "      <td>61</td>\n",
              "      <td>1617</td>\n",
              "      <td>88149</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.134233</td>\n",
              "      <td>25</td>\n",
              "      <td>0.188503</td>\n",
              "      <td>3.721</td>\n",
              "      <td>2</td>\n",
              "      <td>83</td>\n",
              "      <td>93</td>\n",
              "      <td>512</td>\n",
              "      <td>748</td>\n",
              "      <td>188</td>\n",
              "      <td>131</td>\n",
              "      <td>191</td>\n",
              "      <td>131</td>\n",
              "      <td>193</td>\n",
              "      <td>837</td>\n",
              "      <td>200</td>\n",
              "      <td>136</td>\n",
              "      <td>80057</td>\n",
              "      <td>82363</td>\n",
              "      <td>211</td>\n",
              "      <td>144</td>\n",
              "      <td>914</td>\n",
              "      <td>218</td>\n",
              "      <td>86726</td>\n",
              "      <td>966</td>\n",
              "      <td>88703</td>\n",
              "      <td>978</td>\n",
              "      <td>90592</td>\n",
              "      <td>152</td>\n",
              "      <td>92422</td>\n",
              "      <td>1012</td>\n",
              "      <td>93942</td>\n",
              "      <td>262</td>\n",
              "      <td>1029</td>\n",
              "      <td>95385</td>\n",
              "      <td>158</td>\n",
              "      <td>1051</td>\n",
              "      <td>97032</td>\n",
              "      <td>270</td>\n",
              "      <td>...</td>\n",
              "      <td>1193</td>\n",
              "      <td>116300</td>\n",
              "      <td>329</td>\n",
              "      <td>1193</td>\n",
              "      <td>329</td>\n",
              "      <td>170</td>\n",
              "      <td>1193</td>\n",
              "      <td>1193</td>\n",
              "      <td>170</td>\n",
              "      <td>1193</td>\n",
              "      <td>116388</td>\n",
              "      <td>329</td>\n",
              "      <td>170</td>\n",
              "      <td>1193</td>\n",
              "      <td>329</td>\n",
              "      <td>170</td>\n",
              "      <td>1193</td>\n",
              "      <td>329</td>\n",
              "      <td>1193</td>\n",
              "      <td>1193</td>\n",
              "      <td>329</td>\n",
              "      <td>1193</td>\n",
              "      <td>116462</td>\n",
              "      <td>1193</td>\n",
              "      <td>1193</td>\n",
              "      <td>170</td>\n",
              "      <td>1193</td>\n",
              "      <td>116521</td>\n",
              "      <td>170</td>\n",
              "      <td>1193</td>\n",
              "      <td>329</td>\n",
              "      <td>170</td>\n",
              "      <td>1193</td>\n",
              "      <td>116570</td>\n",
              "      <td>170</td>\n",
              "      <td>116582</td>\n",
              "      <td>329</td>\n",
              "      <td>170</td>\n",
              "      <td>1193</td>\n",
              "      <td>116597</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.072066</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>25</td>\n",
              "      <td>0.098545</td>\n",
              "      <td>3.851</td>\n",
              "      <td>89140</td>\n",
              "      <td>402</td>\n",
              "      <td>534</td>\n",
              "      <td>8199</td>\n",
              "      <td>27300</td>\n",
              "      <td>96233</td>\n",
              "      <td>1805</td>\n",
              "      <td>113261</td>\n",
              "      <td>2058</td>\n",
              "      <td>130061</td>\n",
              "      <td>44363</td>\n",
              "      <td>162042</td>\n",
              "      <td>2812</td>\n",
              "      <td>5635052</td>\n",
              "      <td>6258731</td>\n",
              "      <td>193661</td>\n",
              "      <td>3297</td>\n",
              "      <td>56316</td>\n",
              "      <td>209293</td>\n",
              "      <td>7561286</td>\n",
              "      <td>64110</td>\n",
              "      <td>8145135</td>\n",
              "      <td>67883</td>\n",
              "      <td>8596340</td>\n",
              "      <td>4280</td>\n",
              "      <td>8931074</td>\n",
              "      <td>73219</td>\n",
              "      <td>9192173</td>\n",
              "      <td>264724</td>\n",
              "      <td>74879</td>\n",
              "      <td>9420445</td>\n",
              "      <td>4624</td>\n",
              "      <td>76255</td>\n",
              "      <td>9645940</td>\n",
              "      <td>276018</td>\n",
              "      <td>...</td>\n",
              "      <td>120799</td>\n",
              "      <td>18559461</td>\n",
              "      <td>463604</td>\n",
              "      <td>120830</td>\n",
              "      <td>463686</td>\n",
              "      <td>7599</td>\n",
              "      <td>120862</td>\n",
              "      <td>120900</td>\n",
              "      <td>7605</td>\n",
              "      <td>120942</td>\n",
              "      <td>18581657</td>\n",
              "      <td>463963</td>\n",
              "      <td>7607</td>\n",
              "      <td>121034</td>\n",
              "      <td>464036</td>\n",
              "      <td>7611</td>\n",
              "      <td>121080</td>\n",
              "      <td>464106</td>\n",
              "      <td>121119</td>\n",
              "      <td>121156</td>\n",
              "      <td>464233</td>\n",
              "      <td>121198</td>\n",
              "      <td>18619570</td>\n",
              "      <td>121244</td>\n",
              "      <td>121288</td>\n",
              "      <td>7624</td>\n",
              "      <td>121330</td>\n",
              "      <td>18638793</td>\n",
              "      <td>7626</td>\n",
              "      <td>121367</td>\n",
              "      <td>464447</td>\n",
              "      <td>7627</td>\n",
              "      <td>121397</td>\n",
              "      <td>18647758</td>\n",
              "      <td>7628</td>\n",
              "      <td>18651565</td>\n",
              "      <td>464591</td>\n",
              "      <td>7633</td>\n",
              "      <td>121472</td>\n",
              "      <td>18655415</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>25</td>\n",
              "      <td>0.109216</td>\n",
              "      <td>4.901</td>\n",
              "      <td>17304</td>\n",
              "      <td>176</td>\n",
              "      <td>207</td>\n",
              "      <td>398</td>\n",
              "      <td>806</td>\n",
              "      <td>1304</td>\n",
              "      <td>442</td>\n",
              "      <td>1478</td>\n",
              "      <td>475</td>\n",
              "      <td>1643</td>\n",
              "      <td>1011</td>\n",
              "      <td>1950</td>\n",
              "      <td>552</td>\n",
              "      <td>150354</td>\n",
              "      <td>161107</td>\n",
              "      <td>2256</td>\n",
              "      <td>594</td>\n",
              "      <td>1110</td>\n",
              "      <td>2390</td>\n",
              "      <td>178109</td>\n",
              "      <td>1154</td>\n",
              "      <td>185161</td>\n",
              "      <td>1174</td>\n",
              "      <td>191816</td>\n",
              "      <td>653</td>\n",
              "      <td>198422</td>\n",
              "      <td>1216</td>\n",
              "      <td>204869</td>\n",
              "      <td>2907</td>\n",
              "      <td>1233</td>\n",
              "      <td>210997</td>\n",
              "      <td>688</td>\n",
              "      <td>1253</td>\n",
              "      <td>216919</td>\n",
              "      <td>3088</td>\n",
              "      <td>...</td>\n",
              "      <td>1512</td>\n",
              "      <td>429299</td>\n",
              "      <td>6484</td>\n",
              "      <td>1512</td>\n",
              "      <td>6488</td>\n",
              "      <td>1029</td>\n",
              "      <td>1512</td>\n",
              "      <td>1512</td>\n",
              "      <td>1030</td>\n",
              "      <td>1513</td>\n",
              "      <td>430823</td>\n",
              "      <td>6521</td>\n",
              "      <td>1030</td>\n",
              "      <td>1513</td>\n",
              "      <td>6529</td>\n",
              "      <td>1030</td>\n",
              "      <td>1513</td>\n",
              "      <td>6532</td>\n",
              "      <td>1513</td>\n",
              "      <td>1513</td>\n",
              "      <td>6541</td>\n",
              "      <td>1514</td>\n",
              "      <td>433204</td>\n",
              "      <td>1515</td>\n",
              "      <td>1515</td>\n",
              "      <td>1032</td>\n",
              "      <td>1516</td>\n",
              "      <td>434279</td>\n",
              "      <td>1033</td>\n",
              "      <td>1517</td>\n",
              "      <td>6573</td>\n",
              "      <td>1033</td>\n",
              "      <td>1517</td>\n",
              "      <td>435447</td>\n",
              "      <td>1035</td>\n",
              "      <td>436017</td>\n",
              "      <td>6597</td>\n",
              "      <td>1035</td>\n",
              "      <td>1517</td>\n",
              "      <td>436438</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1815</th>\n",
              "      <td>0.000120</td>\n",
              "      <td>0.639053</td>\n",
              "      <td>29</td>\n",
              "      <td>0.108540</td>\n",
              "      <td>4.698</td>\n",
              "      <td>2365</td>\n",
              "      <td>18</td>\n",
              "      <td>22</td>\n",
              "      <td>173</td>\n",
              "      <td>269</td>\n",
              "      <td>164</td>\n",
              "      <td>35</td>\n",
              "      <td>166</td>\n",
              "      <td>36</td>\n",
              "      <td>169</td>\n",
              "      <td>284</td>\n",
              "      <td>181</td>\n",
              "      <td>39</td>\n",
              "      <td>12994</td>\n",
              "      <td>13305</td>\n",
              "      <td>194</td>\n",
              "      <td>41</td>\n",
              "      <td>301</td>\n",
              "      <td>199</td>\n",
              "      <td>13982</td>\n",
              "      <td>313</td>\n",
              "      <td>14250</td>\n",
              "      <td>316</td>\n",
              "      <td>14497</td>\n",
              "      <td>42</td>\n",
              "      <td>14497</td>\n",
              "      <td>326</td>\n",
              "      <td>14497</td>\n",
              "      <td>221</td>\n",
              "      <td>328</td>\n",
              "      <td>15048</td>\n",
              "      <td>42</td>\n",
              "      <td>331</td>\n",
              "      <td>15187</td>\n",
              "      <td>223</td>\n",
              "      <td>...</td>\n",
              "      <td>351</td>\n",
              "      <td>17033</td>\n",
              "      <td>242</td>\n",
              "      <td>351</td>\n",
              "      <td>242</td>\n",
              "      <td>45</td>\n",
              "      <td>351</td>\n",
              "      <td>351</td>\n",
              "      <td>45</td>\n",
              "      <td>351</td>\n",
              "      <td>17035</td>\n",
              "      <td>242</td>\n",
              "      <td>45</td>\n",
              "      <td>351</td>\n",
              "      <td>242</td>\n",
              "      <td>45</td>\n",
              "      <td>351</td>\n",
              "      <td>242</td>\n",
              "      <td>351</td>\n",
              "      <td>351</td>\n",
              "      <td>242</td>\n",
              "      <td>351</td>\n",
              "      <td>17037</td>\n",
              "      <td>351</td>\n",
              "      <td>351</td>\n",
              "      <td>45</td>\n",
              "      <td>351</td>\n",
              "      <td>17040</td>\n",
              "      <td>45</td>\n",
              "      <td>351</td>\n",
              "      <td>242</td>\n",
              "      <td>45</td>\n",
              "      <td>351</td>\n",
              "      <td>17041</td>\n",
              "      <td>45</td>\n",
              "      <td>17043</td>\n",
              "      <td>242</td>\n",
              "      <td>45</td>\n",
              "      <td>351</td>\n",
              "      <td>17044</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1816</th>\n",
              "      <td>0.367586</td>\n",
              "      <td>0.206693</td>\n",
              "      <td>29</td>\n",
              "      <td>0.137349</td>\n",
              "      <td>4.425</td>\n",
              "      <td>6627</td>\n",
              "      <td>40</td>\n",
              "      <td>47</td>\n",
              "      <td>147</td>\n",
              "      <td>231</td>\n",
              "      <td>143</td>\n",
              "      <td>91</td>\n",
              "      <td>151</td>\n",
              "      <td>98</td>\n",
              "      <td>159</td>\n",
              "      <td>281</td>\n",
              "      <td>183</td>\n",
              "      <td>106</td>\n",
              "      <td>34609</td>\n",
              "      <td>36342</td>\n",
              "      <td>213</td>\n",
              "      <td>112</td>\n",
              "      <td>312</td>\n",
              "      <td>232</td>\n",
              "      <td>40315</td>\n",
              "      <td>337</td>\n",
              "      <td>42318</td>\n",
              "      <td>352</td>\n",
              "      <td>44235</td>\n",
              "      <td>134</td>\n",
              "      <td>46090</td>\n",
              "      <td>377</td>\n",
              "      <td>47350</td>\n",
              "      <td>344</td>\n",
              "      <td>384</td>\n",
              "      <td>47911</td>\n",
              "      <td>142</td>\n",
              "      <td>391</td>\n",
              "      <td>48628</td>\n",
              "      <td>357</td>\n",
              "      <td>...</td>\n",
              "      <td>490</td>\n",
              "      <td>68461</td>\n",
              "      <td>511</td>\n",
              "      <td>490</td>\n",
              "      <td>511</td>\n",
              "      <td>163</td>\n",
              "      <td>490</td>\n",
              "      <td>490</td>\n",
              "      <td>163</td>\n",
              "      <td>490</td>\n",
              "      <td>68485</td>\n",
              "      <td>511</td>\n",
              "      <td>163</td>\n",
              "      <td>490</td>\n",
              "      <td>511</td>\n",
              "      <td>163</td>\n",
              "      <td>490</td>\n",
              "      <td>511</td>\n",
              "      <td>490</td>\n",
              "      <td>490</td>\n",
              "      <td>511</td>\n",
              "      <td>490</td>\n",
              "      <td>68545</td>\n",
              "      <td>490</td>\n",
              "      <td>491</td>\n",
              "      <td>163</td>\n",
              "      <td>491</td>\n",
              "      <td>68589</td>\n",
              "      <td>163</td>\n",
              "      <td>491</td>\n",
              "      <td>511</td>\n",
              "      <td>163</td>\n",
              "      <td>491</td>\n",
              "      <td>68614</td>\n",
              "      <td>163</td>\n",
              "      <td>68623</td>\n",
              "      <td>511</td>\n",
              "      <td>163</td>\n",
              "      <td>491</td>\n",
              "      <td>68636</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1817</th>\n",
              "      <td>0.000417</td>\n",
              "      <td>0.743414</td>\n",
              "      <td>29</td>\n",
              "      <td>0.153232</td>\n",
              "      <td>4.351</td>\n",
              "      <td>2381</td>\n",
              "      <td>11</td>\n",
              "      <td>13</td>\n",
              "      <td>95</td>\n",
              "      <td>118</td>\n",
              "      <td>12</td>\n",
              "      <td>20</td>\n",
              "      <td>12</td>\n",
              "      <td>21</td>\n",
              "      <td>12</td>\n",
              "      <td>127</td>\n",
              "      <td>14</td>\n",
              "      <td>24</td>\n",
              "      <td>12428</td>\n",
              "      <td>12656</td>\n",
              "      <td>15</td>\n",
              "      <td>26</td>\n",
              "      <td>143</td>\n",
              "      <td>16</td>\n",
              "      <td>13181</td>\n",
              "      <td>149</td>\n",
              "      <td>13416</td>\n",
              "      <td>150</td>\n",
              "      <td>13575</td>\n",
              "      <td>28</td>\n",
              "      <td>13711</td>\n",
              "      <td>154</td>\n",
              "      <td>13827</td>\n",
              "      <td>16</td>\n",
              "      <td>155</td>\n",
              "      <td>13915</td>\n",
              "      <td>29</td>\n",
              "      <td>157</td>\n",
              "      <td>14099</td>\n",
              "      <td>17</td>\n",
              "      <td>...</td>\n",
              "      <td>169</td>\n",
              "      <td>15200</td>\n",
              "      <td>21</td>\n",
              "      <td>169</td>\n",
              "      <td>21</td>\n",
              "      <td>29</td>\n",
              "      <td>169</td>\n",
              "      <td>169</td>\n",
              "      <td>29</td>\n",
              "      <td>169</td>\n",
              "      <td>15202</td>\n",
              "      <td>21</td>\n",
              "      <td>29</td>\n",
              "      <td>169</td>\n",
              "      <td>21</td>\n",
              "      <td>29</td>\n",
              "      <td>169</td>\n",
              "      <td>21</td>\n",
              "      <td>169</td>\n",
              "      <td>169</td>\n",
              "      <td>21</td>\n",
              "      <td>169</td>\n",
              "      <td>15207</td>\n",
              "      <td>169</td>\n",
              "      <td>169</td>\n",
              "      <td>29</td>\n",
              "      <td>169</td>\n",
              "      <td>15212</td>\n",
              "      <td>29</td>\n",
              "      <td>169</td>\n",
              "      <td>21</td>\n",
              "      <td>29</td>\n",
              "      <td>169</td>\n",
              "      <td>15217</td>\n",
              "      <td>29</td>\n",
              "      <td>15217</td>\n",
              "      <td>21</td>\n",
              "      <td>29</td>\n",
              "      <td>169</td>\n",
              "      <td>15217</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1818</th>\n",
              "      <td>0.172704</td>\n",
              "      <td>0.380792</td>\n",
              "      <td>29</td>\n",
              "      <td>0.132556</td>\n",
              "      <td>3.431</td>\n",
              "      <td>7955</td>\n",
              "      <td>47</td>\n",
              "      <td>55</td>\n",
              "      <td>144</td>\n",
              "      <td>215</td>\n",
              "      <td>69</td>\n",
              "      <td>93</td>\n",
              "      <td>76</td>\n",
              "      <td>96</td>\n",
              "      <td>84</td>\n",
              "      <td>252</td>\n",
              "      <td>90</td>\n",
              "      <td>107</td>\n",
              "      <td>34666</td>\n",
              "      <td>35335</td>\n",
              "      <td>95</td>\n",
              "      <td>111</td>\n",
              "      <td>274</td>\n",
              "      <td>98</td>\n",
              "      <td>36882</td>\n",
              "      <td>290</td>\n",
              "      <td>37483</td>\n",
              "      <td>298</td>\n",
              "      <td>38115</td>\n",
              "      <td>115</td>\n",
              "      <td>38527</td>\n",
              "      <td>306</td>\n",
              "      <td>39039</td>\n",
              "      <td>106</td>\n",
              "      <td>308</td>\n",
              "      <td>39763</td>\n",
              "      <td>118</td>\n",
              "      <td>310</td>\n",
              "      <td>40156</td>\n",
              "      <td>107</td>\n",
              "      <td>...</td>\n",
              "      <td>368</td>\n",
              "      <td>54358</td>\n",
              "      <td>135</td>\n",
              "      <td>368</td>\n",
              "      <td>135</td>\n",
              "      <td>136</td>\n",
              "      <td>368</td>\n",
              "      <td>368</td>\n",
              "      <td>136</td>\n",
              "      <td>368</td>\n",
              "      <td>54380</td>\n",
              "      <td>135</td>\n",
              "      <td>136</td>\n",
              "      <td>368</td>\n",
              "      <td>135</td>\n",
              "      <td>136</td>\n",
              "      <td>368</td>\n",
              "      <td>135</td>\n",
              "      <td>368</td>\n",
              "      <td>368</td>\n",
              "      <td>135</td>\n",
              "      <td>368</td>\n",
              "      <td>54452</td>\n",
              "      <td>368</td>\n",
              "      <td>368</td>\n",
              "      <td>136</td>\n",
              "      <td>368</td>\n",
              "      <td>54488</td>\n",
              "      <td>136</td>\n",
              "      <td>368</td>\n",
              "      <td>135</td>\n",
              "      <td>136</td>\n",
              "      <td>368</td>\n",
              "      <td>54503</td>\n",
              "      <td>136</td>\n",
              "      <td>54513</td>\n",
              "      <td>135</td>\n",
              "      <td>136</td>\n",
              "      <td>368</td>\n",
              "      <td>54518</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1819</th>\n",
              "      <td>0.002834</td>\n",
              "      <td>0.438795</td>\n",
              "      <td>29</td>\n",
              "      <td>0.122905</td>\n",
              "      <td>4.078</td>\n",
              "      <td>2204</td>\n",
              "      <td>15</td>\n",
              "      <td>16</td>\n",
              "      <td>225</td>\n",
              "      <td>464</td>\n",
              "      <td>320</td>\n",
              "      <td>27</td>\n",
              "      <td>346</td>\n",
              "      <td>28</td>\n",
              "      <td>368</td>\n",
              "      <td>580</td>\n",
              "      <td>410</td>\n",
              "      <td>31</td>\n",
              "      <td>19461</td>\n",
              "      <td>20202</td>\n",
              "      <td>450</td>\n",
              "      <td>34</td>\n",
              "      <td>632</td>\n",
              "      <td>467</td>\n",
              "      <td>21644</td>\n",
              "      <td>661</td>\n",
              "      <td>22273</td>\n",
              "      <td>676</td>\n",
              "      <td>22697</td>\n",
              "      <td>41</td>\n",
              "      <td>23258</td>\n",
              "      <td>695</td>\n",
              "      <td>24084</td>\n",
              "      <td>523</td>\n",
              "      <td>704</td>\n",
              "      <td>24590</td>\n",
              "      <td>45</td>\n",
              "      <td>711</td>\n",
              "      <td>24904</td>\n",
              "      <td>536</td>\n",
              "      <td>...</td>\n",
              "      <td>775</td>\n",
              "      <td>29723</td>\n",
              "      <td>638</td>\n",
              "      <td>775</td>\n",
              "      <td>638</td>\n",
              "      <td>49</td>\n",
              "      <td>775</td>\n",
              "      <td>775</td>\n",
              "      <td>49</td>\n",
              "      <td>775</td>\n",
              "      <td>29736</td>\n",
              "      <td>638</td>\n",
              "      <td>49</td>\n",
              "      <td>775</td>\n",
              "      <td>638</td>\n",
              "      <td>49</td>\n",
              "      <td>775</td>\n",
              "      <td>638</td>\n",
              "      <td>775</td>\n",
              "      <td>775</td>\n",
              "      <td>638</td>\n",
              "      <td>775</td>\n",
              "      <td>29792</td>\n",
              "      <td>775</td>\n",
              "      <td>775</td>\n",
              "      <td>49</td>\n",
              "      <td>775</td>\n",
              "      <td>29819</td>\n",
              "      <td>49</td>\n",
              "      <td>775</td>\n",
              "      <td>638</td>\n",
              "      <td>49</td>\n",
              "      <td>775</td>\n",
              "      <td>29831</td>\n",
              "      <td>49</td>\n",
              "      <td>29836</td>\n",
              "      <td>638</td>\n",
              "      <td>49</td>\n",
              "      <td>775</td>\n",
              "      <td>29841</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1820 rows × 326 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "      dominant_color.histogram[4]  ...  views.168\n",
              "0                        0.000000  ...     351137\n",
              "1                        0.000000  ...      88149\n",
              "2                        0.000000  ...     116597\n",
              "3                        0.072066  ...   18655415\n",
              "4                        0.000000  ...     436438\n",
              "...                           ...  ...        ...\n",
              "1815                     0.000120  ...      17044\n",
              "1816                     0.367586  ...      68636\n",
              "1817                     0.000417  ...      15217\n",
              "1818                     0.172704  ...      54518\n",
              "1819                     0.002834  ...      29841\n",
              "\n",
              "[1820 rows x 326 columns]"
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
        "id": "7kBzBe3P25EF",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y_pso = data[\"views.168\"]\n",
        "allowed_rows = 500"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "srEd40GY_ltq",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y_train = data.iloc[0:allowed_rows,data.shape[1]-1].values"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0mVvV1wx3tU5",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "x_pso = data.drop(\"views.168\",axis=1)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "y4-NY4JA_w0m",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "x_train = data.iloc[0:allowed_rows,1:data.shape[1]-1].values"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8S3P5JjRFzgv",
        "colab_type": "code",
        "outputId": "91e53229-2fb4-4373-e527-87b966fcff7d",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 247
        }
      },
      "source": [
        "x_train"
      ],
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[9.87884440e-02, 2.50000000e+01, 1.61409578e-01, ...,\n",
              "        4.91600000e+03, 3.76000000e+02, 4.22500000e+03],\n",
              "       [2.78481010e-02, 2.50000000e+01, 1.06315749e-01, ...,\n",
              "        5.79000000e+02, 6.10000000e+01, 1.61700000e+03],\n",
              "       [1.34232955e-01, 2.50000000e+01, 1.88502919e-01, ...,\n",
              "        3.29000000e+02, 1.70000000e+02, 1.19300000e+03],\n",
              "       ...,\n",
              "       [0.00000000e+00, 2.90000000e+01, 1.16591296e-01, ...,\n",
              "        2.20000000e+01, 6.00000000e+00, 5.60000000e+01],\n",
              "       [9.69362130e-02, 2.90000000e+01, 8.84555440e-02, ...,\n",
              "        3.20000000e+01, 1.10000000e+01, 1.21000000e+02],\n",
              "       [1.72541744e-01, 2.90000000e+01, 9.86650560e-02, ...,\n",
              "        1.30000000e+01, 9.00000000e+00, 3.10000000e+01]])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 40
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dd5MWjBY9ydV",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from sklearn.model_selection import GridSearchCV"
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
        "from sklearn.preprocessing import StandardScaler\n",
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
        "id": "tRNwLf8WvqC6",
        "colab_type": "code",
        "outputId": "ee9be9f5-15bc-4c2b-d026-5a01ae3f5c6b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "y.shape"
      ],
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(500,)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 45
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zwXOQaMHT0vN",
        "colab_type": "code",
        "outputId": "1bd8bf37-b1d7-488f-b0ba-77c8db1e4300",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "x.shape"
      ],
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(500, 324)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 46
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "UsMLbbjDHh0y",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from sklearn.svm import SVR"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "D6jAz95iIHzc",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "svr = SVR(kernel='rbf',C = 10000)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab_type": "code",
        "id": "jZ1rFQq-rVn6",
        "colab": {}
      },
      "source": [
        "clf = GridSearchCV(svr, parameters)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PxUC3r7aJrys",
        "colab_type": "code",
        "outputId": "b4263a04-191c-49a5-fd63-010c5ab5158b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 176
        }
      },
      "source": [
        "clf.fit(x, y)\n",
        "GridSearchCV(estimator=svr,\n",
        "             param_grid={'C': [1, 10000], 'epsilon': [0,2]})"
      ],
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "GridSearchCV(cv=None, error_score=nan,\n",
              "             estimator=SVR(C=10000, cache_size=200, coef0=0.0, degree=3,\n",
              "                           epsilon=0.1, gamma='scale', kernel='rbf',\n",
              "                           max_iter=-1, shrinking=True, tol=0.001,\n",
              "                           verbose=False),\n",
              "             iid='deprecated', n_jobs=None,\n",
              "             param_grid={'C': [1, 10000], 'epsilon': [0, 2]},\n",
              "             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,\n",
              "             scoring=None, verbose=0)"
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
        "id": "u7OqE9mQJ8Dz",
        "colab_type": "code",
        "outputId": "ad4d3c31-098e-4398-d876-6e0060a08498",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "clf.best_params_"
      ],
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'C': 1, 'epsilon': 0}"
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
        "id": "VzstEvxWBogD",
        "colab_type": "code",
        "outputId": "800c3412-0e8a-4879-83a5-b6457a45258a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "score = np.mean(cross_val_score(svr, x, y,  cv=5,  scoring=None))\n",
        "score"
      ],
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.3420750459488038"
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
        "id": "lVYlbkMGaHjq",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "cf422ee2-83af-4cae-baee-c2e32787e29a"
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n",
        "regr = SVR()\n",
        "svm = regr.fit(X_train, y_train)\n",
        "svm.score(X_test,y_test)"
      ],
      "execution_count": 78,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.6989835953785764"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 78
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "snrFuYafBfGL",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        " class GeneticSelector :\n",
        "  \n",
        "  \n",
        "  def __init__(self, estimator, n_gen, size, n_best, n_rand, \n",
        "                 n_children, mutation_rate):\n",
        "    # Estimator \n",
        "    self.estimator = estimator\n",
        "    # Number of generations\n",
        "    self.n_gen = n_gen\n",
        "    # Number of chromosomes in population\n",
        "    self.size = size\n",
        "    # Number of best chromosomes to select\n",
        "    self.n_best = n_best\n",
        "    # Number of random chromosomes to select\n",
        "    self.n_rand = n_rand\n",
        "    # Number of children created during crossover\n",
        "    self.n_children = n_children\n",
        "    # Probablity of chromosome mutation\n",
        "    self.mutation_rate = mutation_rate\n",
        "    if int((self.n_best + self.n_rand) / 2) * self.n_children != self.size:\n",
        "      raise ValueError(\"The population size is not stable.\")\n",
        "  \n",
        "  \n",
        "  def initilize(self):\n",
        "    population = []\n",
        "    for i in range(self.size):\n",
        "      chromosome = np.ones(self.n_features, dtype=np.bool)\n",
        "      mask = np.random.rand(len(chromosome)) < 0.3 #The probability 0.3 is chosen arbitrarily, however it is suggested to avoid large probabilities. We would not like to create chromosomes with all variables excluded\n",
        "      chromosome[mask] = False\n",
        "      population.append(chromosome)\n",
        "    return population\n",
        "  \n",
        "  \n",
        "  def fitness(self, population):\n",
        "    X, y = self.dataset\n",
        "    scores = []\n",
        "    for chromosome in population:\n",
        "      score =-1* np.mean(cross_val_score(self.estimator, X[:,chromosome], y, \n",
        "                                                        cv=5, \n",
        "                                                        scoring=None))\n",
        "      scores.append(score)\n",
        "    scores, population = np.array(scores), np.array(population) \n",
        "    inds = np.argsort(scores)\n",
        "    print(-1*scores[inds[0]])\n",
        "    #print(population[inds[0]])\n",
        "    return list(scores[inds]), list(population[inds,:])\n",
        "  \n",
        "  \n",
        "  def select(self, population_sorted):\n",
        "    population_next = []\n",
        "    for i in range(self.n_best):\n",
        "      population_next.append(population_sorted[i])\n",
        "    for i in range(self.n_rand):\n",
        "      population_next.append(random.choice(population_sorted))\n",
        "    random.shuffle(population_next)\n",
        "    return population_next\n",
        "  \n",
        "  \n",
        "  def crossover(self, population):\n",
        "    population_next = []\n",
        "    for i in range(int(len(population)/2)):\n",
        "      for j in range(self.n_children):\n",
        "        #print(i)\n",
        "        chromosome1, chromosome2 = population[i], population[len(population)-1-j]\n",
        "        #print(type(chromosome2))\n",
        "        child = chromosome1\n",
        "        mask = np.random.rand(len(child)) > 0.5\n",
        "        for k in range(len(mask)):\n",
        "          if(mask[k] == True):\n",
        "            child[k]=chromosome2[k]\n",
        "        population_next.append(child)\n",
        "    return population_next\n",
        "  \n",
        "  \n",
        "  def mutate(self, population):\n",
        "    population_next = []\n",
        "    for i in range(len(population)):\n",
        "      chromosome = population[i]\n",
        "      if random.random() < self.mutation_rate:\n",
        "        mask = np.random.rand(len(chromosome)) < 0.05\n",
        "        for k in range(len(mask)):\n",
        "          if(mask[k]==True):\n",
        "            chromosome[k] = False\n",
        "      population_next.append(chromosome)\n",
        "    return population_next  \n",
        "  \n",
        "  \n",
        "  def generate(self, population):\n",
        "    # Selection, crossover and mutation\n",
        "    scores_sorted, population_sorted = self.fitness(population)\n",
        "    #print(type(population_sorted[1]))\n",
        "    population = self.select(population_sorted)\n",
        "    #print(type(population[3]))\n",
        "    population = self.crossover(population)\n",
        "    #print(len(population))\n",
        "    population = self.mutate(population)\n",
        "    #print(len(population))\n",
        "    # History\n",
        "    self.chromosomes_best.append(population_sorted[0])\n",
        "    self.scores_best.append(scores_sorted[0])\n",
        "    self.scores_avg.append(np.mean(scores_sorted))\n",
        "    return population\n",
        "  \n",
        "  \n",
        "  def fit(self, X, y):\n",
        "    self.chromosomes_best = []\n",
        "    self.scores_best, self.scores_avg  = [], []\n",
        "    self.dataset = X, y\n",
        "    self.n_features = X.shape[1]\n",
        "    population = self.initilize()\n",
        "    for i in range(self.n_gen):\n",
        "      population = self.generate(population)\n",
        "    return self \n",
        "    \n",
        "  @property\n",
        "  \n",
        "  \n",
        "  def support_(self):\n",
        "      return self.chromosomes_best[-1]\n",
        " \n",
        "  \n",
        "  def plot_scores(self):\n",
        "      plt.plot(self.scores_best, label='Best')\n",
        "      plt.plot(self.scores_avg, label='Average')\n",
        "      plt.legend()\n",
        "      plt.ylabel('Scores')\n",
        "      plt.xlabel('Generation')\n",
        "      plt.show()\n",
        "  \n",
        "  def ret_pop(self):\n",
        "    return self.chromosomes_best \n",
        "  "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rejhSRnSS-z7",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def initilize(self):\n",
        "  population = []\n",
        "  for i in range(self.size):\n",
        "    chromosome = np.ones(self.n_features, dtype=np.bool)\n",
        "    mask = np.random.rand(len(chromosome)) < 0.3 #The probability 0.3 is chosen arbitrarily, however it is suggested to avoid large probabilities. We would not like to create chromosomes with all variables excluded\n",
        "    chromosome[mask] = False\n",
        "    population.append(chromosome)\n",
        "    return population"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AHzASH7rTq2j",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def fitness(self, population):\n",
        "  X, y = self.dataset\n",
        "  scores = []\n",
        "  for chromosome in population:\n",
        "    score = -1.0 * np.mean(cross_val_score(self.estimator, X[:,chromosome], y, \n",
        "                                                        cv=5, \n",
        "                                                        scoring=\"neg_mean_squared_error\"))\n",
        "    scores.append(score)\n",
        "  scores, population = np.array(scores), np.array(population) \n",
        "  inds = np.argsort(scores)\n",
        "  print(scores[-1])\n",
        "  return list(scores[inds]), list(population[inds,:])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wuYZWDcDVZ6G",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def select(self, population_sorted):\n",
        "  population_next = []\n",
        "  for i in range(self.n_best):\n",
        "    population_next.append(population_sorted)\n",
        "  for i in range(self.n_rand):\n",
        "    population_next.append(random.choice(population_sorted))\n",
        "  random.shuffle(population_next)\n",
        "  return population_next\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vgfQ7ZoXViNX",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def crossover(self, population):\n",
        "  population_next = []\n",
        "  for i in range(int(len(population)/2)):\n",
        "    for j in range(self.n_children):\n",
        "      chromosome1, chromosome2 = population[i], population[len(population)-1-i]\n",
        "      child = chromosome1\n",
        "      mask = np.random.rand(len(child)) > 0.5\n",
        "      child[mask] = chromosome2[mask]\n",
        "      population_next.append(child)\n",
        "  return population_next"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wwU27BGETQhk",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def mutate(self, population):\n",
        "  population_next = []\n",
        "  for i in range(len(population)):\n",
        "    chromosome = population[i]\n",
        "    if random.random() < self.mutation_rate:\n",
        "      mask = np.random.rand(len(chromosome)) < 0.05\n",
        "      chromosome[mask] = False\n",
        "      population_next.append(chromosome)\n",
        "  return population_next"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7TzBfXYNT6OC",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def generate(self, population):\n",
        "# Selection, crossover and mutation\n",
        "  scores_sorted, population_sorted = self.fitness(population)\n",
        "  population = self.select(population_sorted)\n",
        "  population = self.crossover(population)\n",
        "  population = self.mutate(population)\n",
        "  # History\n",
        "  self.chromosomes_best.append(population_sorted[0])\n",
        "  self.scores_best.append(scores_sorted[0])\n",
        "  self.scores_avg.append(np.mean(scores_sorted))\n",
        "  return population"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "opTJL4KrYehZ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def fit(self, X, y):\n",
        "  self.chromosomes_best = []\n",
        "  self.scores_best, self.scores_avg  = [], []\n",
        "  self.dataset = x_train, y_train\n",
        "  self.n_features = x_train.shape[1]\n",
        "  population = self.initilize()\n",
        "  for i in range(self.n_gen):\n",
        "    population = self.generate(population)\n",
        "    return self \n",
        "    \n",
        "  @property\n",
        "  def support_(self):\n",
        "    return self.chromosomes_best[-1]\n",
        " \n",
        "  def plot_scores(self):\n",
        "    plt.plot(-1*self.scores_best, label='Best')\n",
        "    plt.plot(-1*self.scores_avg, label='Average')\n",
        "    plt.legend()\n",
        "    plt.ylabel('Scores')\n",
        "    plt.xlabel('Generation')\n",
        "    plt.show()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rlKIIz2kZspq",
        "colab_type": "code",
        "outputId": "630f5f51-da66-487b-8a26-e170f4e55a25",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 474
        }
      },
      "source": [
        "selector = GeneticSelector(estimator=svr, \n",
        "                      n_gen=10, size=100, n_best=20, n_rand=20, \n",
        "                      n_children=5, mutation_rate=0.05)\n",
        "selector.fit(x, y)\n",
        "selector.plot_scores()\n",
        "score = cross_val_score(svr, x[:,selector.support_], y, cv=5, scoring=None)\n",
        "print(\"Score after feature selection: {:.2f}\".format(np.mean(score)))"
      ],
      "execution_count": 62,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0.3868360010140915\n",
            "0.3868605846457965\n",
            "0.4028724929334427\n",
            "0.39927380688357195\n",
            "0.4127828748503788\n",
            "0.41474722687634574\n",
            "0.4178019593133782\n",
            "0.41995836883083904\n",
            "0.4196366204613115\n",
            "0.42255285009904037\n"
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
            "Score after feature selection: 0.42\n"
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
        "outputId": "3ad59198-42f7-4219-b94c-2e7cce9c197e",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 278
        }
      },
      "source": [
        "populati=selector.ret_pop()\n",
        "\n",
        "selector.plot_scores()"
      ],
      "execution_count": 63,
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
        "id": "RwDhIS3-zusr",
        "colab_type": "code",
        "outputId": "c60dc721-905e-4dbe-b6dd-df0df05873e4",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "len(populati)\n"
      ],
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "10"
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
        "id": "YPM1utGH6c5G",
        "colab_type": "code",
        "outputId": "7398458b-8638-4694-f3df-1c54c7e6a34c",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 194
        }
      },
      "source": [
        "for i in range(len(populati)):\n",
        "  count=0\n",
        "  for j in range(len(x[0])):\n",
        "    if(populati[i][j]==True):\n",
        "      count=count+1\n",
        "  print(\"features selected in \",(i+1),\" th generation =\",count)\n"
      ],
      "execution_count": 65,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "features selected in  1  th generation = 215\n",
            "features selected in  2  th generation = 224\n",
            "features selected in  3  th generation = 232\n",
            "features selected in  4  th generation = 221\n",
            "features selected in  5  th generation = 214\n",
            "features selected in  6  th generation = 200\n",
            "features selected in  7  th generation = 189\n",
            "features selected in  8  th generation = 194\n",
            "features selected in  9  th generation = 200\n",
            "features selected in  10  th generation = 201\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TIru1URLigKy",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "score = np.mean(cross_val_score(svr, x[:,populati[0]], y, cv=5, scoring=None))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SmZMAMdrccnP",
        "colab_type": "code",
        "outputId": "37502dfe-2807-4f2e-ac2b-b3e2e6ca3671",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "score"
      ],
      "execution_count": 67,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.3868360010140915"
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
        "id": "UoOpHrTz4LiX",
        "colab_type": "code",
        "outputId": "72d82e41-7a1e-452e-8443-46d6ade5e7f5",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "x_train.shape"
      ],
      "execution_count": 68,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(500, 324)"
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
        "outputId": "01c27b4f-69fa-41d3-a8a1-33a4f585b6e8",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        }
      },
      "source": [
        "sel = GeneticSelector(estimator=svr, \n",
        "                      n_gen=50, size=200, n_best=40, n_rand=40, \n",
        "                      n_children=5, mutation_rate=0.05)\n",
        "sel.fit(x, y.ravel())\n",
        "sel.plot_scores()\n",
        "score = cross_val_score(svr, x[:,sel.support_], y, cv=5, scoring=None)\n",
        "print(\"Score after feature selection: {:.2f}\".format(np.mean(score)))"
      ],
      "execution_count": 71,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0.3914714092403696\n",
            "0.3814298401241686\n",
            "0.3847104115633215\n",
            "0.3830253692412269\n",
            "0.38736058256891\n",
            "0.39466880928990555\n",
            "0.38759423525542014\n",
            "0.3895177790274825\n",
            "0.39286614738138526\n",
            "0.401339088147307\n",
            "0.3979192559455113\n",
            "0.4053152910719785\n",
            "0.40778711298157616\n",
            "0.39931707079359546\n",
            "0.4000132440470849\n",
            "0.4064851576208321\n",
            "0.3992838306228253\n",
            "0.4096583129914764\n",
            "0.41430866475928746\n",
            "0.41696254250093967\n",
            "0.4238313762246765\n",
            "0.43615041630824286\n",
            "0.44254287661970365\n",
            "0.4365986560005071\n",
            "0.42564553643425296\n",
            "0.43095201725491245\n",
            "0.45379995702596065\n",
            "0.4529183062301915\n",
            "0.4619347197138778\n",
            "0.4571637533721319\n",
            "0.4616653191714709\n",
            "0.4682114453274779\n",
            "0.4645497137451562\n",
            "0.4712670554474518\n",
            "0.49469721321646076\n",
            "0.48832858002688334\n",
            "0.48069522590219993\n",
            "0.4834717246125272\n",
            "0.5034835272101985\n",
            "0.49003265369457605\n",
            "0.5012904046375303\n",
            "0.501555971217571\n",
            "0.4978374829126473\n",
            "0.498689984940296\n",
            "0.5412826374371023\n",
            "0.5270038268243337\n",
            "0.5204672892012348\n",
            "0.5074162231164225\n",
            "0.538802933354659\n",
            "0.5250589814848619\n"
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
            "Score after feature selection: 0.52\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zPi_StgP6pWm",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "lst = []\n",
        "columns = list(data.columns)\n",
        "support = list(sel.support_)\n",
        "for i in range(len(support)):\n",
        "  if support[i] == False:\n",
        "    lst.append(columns[i]) "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DU6SPM3nB-pH",
        "colab_type": "code",
        "outputId": "a7ae7a9c-d813-4d5d-8115-b8d4d9b642fd",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        }
      },
      "source": [
        "lst"
      ],
      "execution_count": 73,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['dominant_color.histogram[9]',\n",
              " 'title.sentiment',\n",
              " 'comments.2',\n",
              " 'comments.3',\n",
              " 'likes.3',\n",
              " 'comments.11',\n",
              " 'shares.12',\n",
              " 'comments.12',\n",
              " 'likes.14',\n",
              " 'views.16',\n",
              " 'views.17',\n",
              " 'shares.17',\n",
              " 'comments.17',\n",
              " 'likes.17',\n",
              " 'shares.18',\n",
              " 'views.19',\n",
              " 'likes.19',\n",
              " 'views.20',\n",
              " 'likes.20',\n",
              " 'views.21',\n",
              " 'comments.21',\n",
              " 'views.22',\n",
              " 'likes.22',\n",
              " 'shares.23',\n",
              " 'likes.23',\n",
              " 'views.24',\n",
              " 'comments.24',\n",
              " 'likes.24',\n",
              " 'views.25',\n",
              " 'shares.25',\n",
              " 'comments.25',\n",
              " 'likes.25',\n",
              " 'shares.26',\n",
              " 'comments.26',\n",
              " 'likes.26',\n",
              " 'shares.27',\n",
              " 'comments.27',\n",
              " 'likes.27',\n",
              " 'views.28',\n",
              " 'likes.28',\n",
              " 'likes.29',\n",
              " 'views.30',\n",
              " 'shares.30',\n",
              " 'views.31',\n",
              " 'shares.31',\n",
              " 'comments.31',\n",
              " 'likes.31',\n",
              " 'shares.32',\n",
              " 'shares.34',\n",
              " 'comments.34',\n",
              " 'likes.34',\n",
              " 'shares.35',\n",
              " 'shares.36',\n",
              " 'comments.36',\n",
              " 'likes.36',\n",
              " 'comments.38',\n",
              " 'likes.38',\n",
              " 'comments.39',\n",
              " 'comments.40',\n",
              " 'likes.40',\n",
              " 'shares.41',\n",
              " 'likes.42',\n",
              " 'views.43',\n",
              " 'shares.43',\n",
              " 'likes.43',\n",
              " 'comments.44',\n",
              " 'shares.45',\n",
              " 'comments.45',\n",
              " 'likes.45',\n",
              " 'shares.46',\n",
              " 'likes.46',\n",
              " 'shares.49',\n",
              " 'comments.49',\n",
              " 'likes.49',\n",
              " 'views.50',\n",
              " 'shares.50',\n",
              " 'comments.50',\n",
              " 'comments.51',\n",
              " 'likes.51',\n",
              " 'shares.52',\n",
              " 'comments.52',\n",
              " 'shares.53',\n",
              " 'views.54',\n",
              " 'comments.54',\n",
              " 'likes.54',\n",
              " 'views.55',\n",
              " 'likes.55',\n",
              " 'views.56',\n",
              " 'shares.56',\n",
              " 'comments.56',\n",
              " 'likes.56',\n",
              " 'views.57',\n",
              " 'shares.58',\n",
              " 'shares.59',\n",
              " 'views.63',\n",
              " 'comments.63',\n",
              " 'likes.63',\n",
              " 'comments.64',\n",
              " 'views.65',\n",
              " 'shares.65',\n",
              " 'views.66',\n",
              " 'likes.66',\n",
              " 'comments.67',\n",
              " 'likes.67',\n",
              " 'comments.68',\n",
              " 'likes.68',\n",
              " 'views.69',\n",
              " 'likes.69',\n",
              " 'comments.72',\n",
              " 'views.74',\n",
              " 'shares.75',\n",
              " 'shares.76',\n",
              " 'likes.76',\n",
              " 'views.77',\n",
              " 'likes.77',\n",
              " 'views.78',\n",
              " 'shares.78',\n",
              " 'views.79',\n",
              " 'views.80',\n",
              " 'shares.80',\n",
              " 'views.81',\n",
              " 'shares.81',\n",
              " 'likes.81',\n",
              " 'views.82',\n",
              " 'comments.83',\n",
              " 'shares.84',\n",
              " 'comments.84',\n",
              " 'shares.85',\n",
              " 'likes.85',\n",
              " 'views.87',\n",
              " 'comments.87',\n",
              " 'shares.89',\n",
              " 'views.92',\n",
              " 'views.93',\n",
              " 'shares.93',\n",
              " 'likes.93',\n",
              " 'comments.94',\n",
              " 'views.95',\n",
              " 'shares.95',\n",
              " 'comments.95',\n",
              " 'shares.97',\n",
              " 'comments.97',\n",
              " 'comments.98',\n",
              " 'views.99',\n",
              " 'comments.99',\n",
              " 'views.100',\n",
              " 'comments.100',\n",
              " 'likes.100',\n",
              " 'comments.101',\n",
              " 'views.102',\n",
              " 'views.103',\n",
              " 'comments.103',\n",
              " 'shares.105',\n",
              " 'comments.106',\n",
              " 'views.107',\n",
              " 'shares.107',\n",
              " 'comments.107',\n",
              " 'likes.107',\n",
              " 'comments.108',\n",
              " 'likes.108',\n",
              " 'likes.109',\n",
              " 'likes.110',\n",
              " 'views.111',\n",
              " 'views.112',\n",
              " 'shares.112',\n",
              " 'comments.112',\n",
              " 'comments.113',\n",
              " 'shares.115',\n",
              " 'comments.115',\n",
              " 'views.116',\n",
              " 'shares.116',\n",
              " 'views.117',\n",
              " 'comments.117',\n",
              " 'likes.117',\n",
              " 'views.118',\n",
              " 'views.119',\n",
              " 'likes.119',\n",
              " 'shares.121',\n",
              " 'likes.121',\n",
              " 'shares.122',\n",
              " 'comments.122',\n",
              " 'comments.125',\n",
              " 'likes.125',\n",
              " 'views.126',\n",
              " 'shares.126',\n",
              " 'comments.126',\n",
              " 'shares.127',\n",
              " 'comments.128',\n",
              " 'shares.129',\n",
              " 'likes.129',\n",
              " 'shares.130',\n",
              " 'likes.130',\n",
              " 'views.131',\n",
              " 'shares.131',\n",
              " 'comments.132',\n",
              " 'comments.133',\n",
              " 'likes.133',\n",
              " 'views.134',\n",
              " 'likes.134',\n",
              " 'likes.135',\n",
              " 'likes.137',\n",
              " 'likes.138',\n",
              " 'shares.140',\n",
              " 'comments.140',\n",
              " 'likes.140',\n",
              " 'views.143',\n",
              " 'views.144',\n",
              " 'likes.145',\n",
              " 'views.146',\n",
              " 'comments.146',\n",
              " 'likes.146',\n",
              " 'shares.147',\n",
              " 'views.148',\n",
              " 'comments.149',\n",
              " 'views.150',\n",
              " 'shares.150',\n",
              " 'likes.150',\n",
              " 'shares.151',\n",
              " 'likes.151',\n",
              " 'shares.152',\n",
              " 'comments.152',\n",
              " 'likes.152',\n",
              " 'comments.154',\n",
              " 'shares.156',\n",
              " 'comments.156',\n",
              " 'likes.156',\n",
              " 'shares.157',\n",
              " 'comments.157',\n",
              " 'likes.157',\n",
              " 'shares.158',\n",
              " 'likes.158',\n",
              " 'likes.159',\n",
              " 'likes.160',\n",
              " 'views.161',\n",
              " 'likes.161',\n",
              " 'likes.162',\n",
              " 'comments.163',\n",
              " 'likes.163',\n",
              " 'comments.164',\n",
              " 'likes.164',\n",
              " 'comments.165',\n",
              " 'likes.165',\n",
              " 'views.166',\n",
              " 'views.167',\n",
              " 'comments.168']"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 73
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1OMcETso6vqE",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "ans = data.drop(lst,axis=1)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_k_sH4LFBlxF",
        "colab_type": "code",
        "outputId": "83a474a1-bd6f-4dc9-8693-47bc18c17d72",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "#X = data.drop('views.168',axis = 1)\n",
        "#y = data['views.168']\n",
        "allowed_rows=1802\n",
        "X = ans.iloc[0:allowed_rows,1:ans.shape[1]-1].values\n",
        "y = ans.iloc[0:allowed_rows,ans.shape[1]-1].values\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "sc_x=StandardScaler()\n",
        "sc_y=StandardScaler()\n",
        "X=sc_x.fit_transform(X)\n",
        "y=sc_y.fit_transform(y.reshape(-1,1))\n",
        "y= y.ravel()\n",
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n",
        "regr = SVR()\n",
        "svm = regr.fit(X_train, y_train)\n",
        "svm.score(X_test,y_test)"
      ],
      "execution_count": 76,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.6989835953785764"
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
        "id": "LqTrmxsKDgLv",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "ans.to_csv('after_ga.csv', index=False)\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vKdvrovqD0_t",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from google.colab import files\n",
        "\n",
        "files.download('after_ga.csv')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8cWcBK_77UbU",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZlS6zip8FsmH",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import numpy as np\n",
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
      "cell_type": "code",
      "metadata": {
        "id": "9_4x-M0lrkpb",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "child"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XnTNaaoUQuVZ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "cgf=[9,8,7]\n",
        "np.argsort(cgf)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nPInmXYoXaEF",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "np.random.rand(900)<.3"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "veh-m95qqkU-",
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