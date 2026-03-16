alpha_c = 0    #for future...keep curcumin at a range of 0.1 to 0.2

Beta = 0.000124   #mins: 0.00012 - 0.000124

# Deltas
S_c = .9 #0.9      #min: 0.9    #hrs: 0.015 
S_p = 9e-6 #9e-6     #min: 9e-6   #hrs: 0.00399995 
S_i = 2 #2   #min: 3.162   #hrs: 0.0527 #can't be anything other than 2 

lambda_p = 18 #18       #min: 18       #hrs: 0.3
lambda_i = .3 #0.3      #min: 0.3      #hrs: 0.005 #can't be more than .4

# mu 
u_c = 1.8 #1.8      #min: 1.8    #hrs: 0.03 

r = 1.8 #1.8             #min: 1.8     #hrs: 0.03 #0 is important 

#reproductive number
r_0 = (u_c * (1 - alpha_c) + Beta * (lambda_p / S_p)) / ((S_c * (1 + alpha_c)) * (lambda_i / S_i)) 



