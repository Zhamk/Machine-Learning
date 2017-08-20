function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
for i = 1:m
    sum=0;
    for k = 1:size(X,2)
       sum=sum+theta(k)*X(i,k); 
    end
    J=J+1/m*(-1*y(i,1)*log(sigmoid(sum))-(1-y(i,1))*log(1-sigmoid(sum)));
end
for j=2:size(X,2)
J=J+lambda/(2*m)*(theta(j)^2);
end
for j =1:size(X,2)
    if j==1
    for i=1:m
        sum=0;
        for k = 1:size(X,2)
           sum=sum+theta(k)*X(i,k); 
        end
    grad(j,1)=grad(j,1)+1/m*(sigmoid(sum)-y(i,1))*X(i,j);
    end
    else
    for i=1:m
        sum=0;
        for k = 1:size(X,2)
           sum=sum+theta(k)*X(i,k); 
        end
    grad(j,1)=grad(j,1)+1/m*(sigmoid(sum)-y(i,1))*X(i,j);
    end
    grad(j,1)=grad(j,1)+lambda/m*theta(j);
    end
end

% =============================================================

end


