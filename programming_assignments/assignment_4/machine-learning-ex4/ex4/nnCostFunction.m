function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
a1=X;
a1=[ones(size(a1,1),1) a1];
z2=a1*Theta1';
a2=sigmoid(z2);
a2=[ones(size(a2,1),1) a2];
z3=a2*Theta2';
a3=sigmoid(z3);

%p is prediction vector
%[p1,p]=max(a3, [], 2);

%recoding vector of 1-9 numbers to matrix of 0' and 1's
Y=zeros(m,num_labels);
%P=Y;
%prediction and y matrices

for i=1:length(y)
Y(i,y(i))=1;
% P(i,p(i))=1;
end
J=(1/m)*sum(sum(((-Y).*(log(a3))-(1-Y).*log(1-a3))));
Theta1_con=Theta1(:,2:end);
Theta2_con=Theta2(:,2:end);
J_regul=lambda/(2*m)*(sum(sum(Theta1_con.*Theta1_con)) +sum(sum(Theta2_con.*Theta2_con)));
J=J+J_regul;



%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

delta_big1=zeros(size(Theta1,1), size(Theta1,2));
delta_big2=zeros(size(Theta2,1), size(Theta2,2));
delta_small3=a3-Y;
delta_small2=delta_small3*Theta2.*(a2.*(1-a2));

%getting rid of the 1st column
delta_small2=delta_small2(:,2:end);

delta_big2=delta_small3'*a2;
delta_big1=delta_small2'*a1;
Theta2_grad=delta_big2/m;
Theta1_grad=delta_big1/m;









% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

regul_2=Theta2*(lambda)/m;
regul_2(:,1)=0;
regul_1=Theta1*(lambda)/m;
regul_1(:,1)=0;
Theta2_grad=delta_big2/m+regul_2;
Theta1_grad=delta_big1/m+regul_1;

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
