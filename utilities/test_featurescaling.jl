include("featurescaling.jl")


X = [randn(100); 5+10*randn(100); 50+2*randn(100)]

X = reshape(X,(100,3))'

X_stand = standardization(X)
mean(X_stand,2)
std(X_stand,2)

standardization!(X)
mean(X,2)
std(X,2)


X = [randn(100); 5+10*randn(100); 50+2*randn(100)]

X = reshape(X,(100,3))

X_stand = standardization(X)
mean(X_stand,2)
std(X_stand,2)

standardization!(X)
mean(X,1)
std(X,1)


X = 5+10*randn(100)

X_stand = standardization(X)
mean(X_stand)
std(X_stand)

standardization!(X)
mean(X)
std(X)
