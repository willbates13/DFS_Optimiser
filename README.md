# DFS_Optimiser

## ⚠️ DISCLAIMER
This repository serves as a public representation of a private Daily Fantasy Sports (DFS) optimization solution. Some scripts have been omitted or rewritten, and the historical dataset includes randomised noise for confidentiality purposes.

## 🚀 Description
This project showcases my expertise in Python by presenting an optimizer for Daily Fantasy Sports (DFS) contests, specifically targeting Draftkings NFL DFS lineups. The optimizer leverages free online projections and calculates optimal lineups based on user-defined risk tolerance.

## 🌐 Introduction
Motivated by the pioneering works of researchers like Hunter et al. (2016) and Bergman et al. (2021), who explored innovative algorithms for winning DFS tournaments, I embarked on creating a novel solution. My approach incorporates stochastic optimization to enhance the chances of success in DFS contests with numerous participants.

## 🔍 Methodology
### Player Projections
I gathered three years' worth of historical DFS data, encompassing projected and actual scores from past NFL players. Additionally, I developed a web scraping script to collect free online projections. Utilizing a similarity algorithm similar to the one in this repository, I identified historically similar examples, allowing me to estimate player variance and covariance. This data-intensive approach enables accurate modeling of player interactions during an NFL weekend, assuming normality of fantasy points, as supported by D. Bergman et al.

### Lineup Optimization
DFS tournaments aim to outperform as many competitors as possible, emphasizing upside (variance) over expected score (mean). With the DFS landscape evolving, players have recognized the effectiveness of "stacking" – selecting players with correlated outcomes. Leveraging statistical insights into player interactions, I calculated precise lineups that maximize the probability of reaching a specified score, considering historical variance and covariance. This results in an accurate, statistically backed form of "stacking". The algorithm utilizes a combination of branch and bound and the PuLP Python library for Mixed-Integer Linear Programming (MILP) to address the optimization problem. The solution within this repository shows how PuLP can be used to create lineups, and allows the user to scale how much they want the variance and covaraince to contribute to the created lineups, giving a simple example of MiLP within Python.

## 📈 Takeaways
Throughout the algorithm development process, I explored various optimization techniques such as neural networks and genetic algorithms. The experience enhanced my confidence in Python and fueled my appetite for future algorithmic problem-solving endeavors.

## 💻 How to Use
The repository is preconfigured for immediate use, with the required data already included. Users interested in applying the model for upcoming contests can refer to the provided script for projection website details and download the salary data in CSV format from the DraftKings website.

# Acknowledgements
David Bergman , Carlos Cardonha, Jason Imbrogno, and Leonardo Lozano. Optimizing the expected maximum of two linear functions defined on a multivariate Gaussian distribution. 2021.
David S. Hunter, Juan Pablo Vielma, and Tauhid Zaman. Picking Winners Using Integer Programming. 2016.
