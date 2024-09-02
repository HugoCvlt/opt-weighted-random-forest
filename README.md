# optimal-weighted-random-forest



![Recap result](img/Tableau_resultat.PNG)

## How to use it ?
-----
### Clone this repository
```bash
git clone https://github.com/HugoCvlt/opt-weighted-random-forest.git
```

### Install requirement
```bash
pip install -r requirements.txt
```

### Exemple

```python
  X_train = pd.read_csv(your_path)
  y_train = pd.read_csv(your_path)
  
  X_val = pd.read_csv(your_path)
  y_val =pd.read_csv(your_path)
  
  Mn = 100
  n=len(X_train)
  n_min = round(np.sqrt(n))
  
  owrf = Opt_WRF(X_train, y_train, Mn=Mn, n_min=n_min)
  
  owrf.two_steps_WRF_opt(verbose=True)
  rf_mae, opt_mae, rf_rmse, opt_rmse = owrf.validate(X_val, y_val)

  print("Random Forest MAE val :", np.array(rf_mae))
  print("Optimal WRF MAE val : ", np.array(opt_mae))
  print("Random Forest RMSE val :", np.array(rf_rmse))
  print("Optimal WRF RMSE val : ", np.array(opt_rmse))
```
