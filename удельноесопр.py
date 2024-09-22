rpr=float(input())
l=float(input())
d=0.04
p=rpr*3.14*(d**2)/(4*l)
print(p)
sigmaR=float(input())
sigmad=0.001
sigmal=0.1
sigma_p=p*((sigmaR/rpr)**2+(2*sigmad/d)**2 + (sigmal/l)**2)**0.5
print(sigma_p)