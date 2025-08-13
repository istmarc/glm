module glm.optim;

import mir.ndslice;
import numir;
import mir.random;
import mir.random.algorithm: randomSlice;
import mir.random.variable: NormalVariable;
import mir.blas;

/++
Abstract class representing gradient descent algotihms
+/
abstract class gradientDescentAlgorithm(T){
   //Slice!(T*) optimize(Slice!(T*) delegate(Slice(const(T)*)) gradF);
};

/++
Gradient descent algorithm.
+/
class gradientDescent(T) : gradientDescentAlgorithm!T {
private:
   size_t n;
   T eps;
   T lr;
   size_t iterMax;

public:
   this(size_t size, T learningRate, T epsilon, size_t nIterMax) {
      n = size;
      lr = learningRate;
      eps = epsilon;
      iterMax = nIterMax;

   }

   // Optimize a function using gradient descent
   Slice!(T*) optimize(Slice!(T*) delegate(Slice!(const(T)*)) gradF) {
      auto rng = Random(unpredictableSeed);
      auto prevXn = randomSlice(rng, NormalVariable!T(), n);
      auto xn = empty!T(n);
      auto diff = empty!T(n);
      foreach(iter; 0..iterMax) {
         auto grad = gradF(prevXn);
         foreach(i; 0..n) {
            xn[i] = prevXn[i] - lr * grad[i];
         }
         // Check for stopping
         foreach(i; 0..n) {
            diff[i] = xn[i] - prevXn[i];
         }
         T norm = nrm2!T(diff);
         if (norm <= eps) {
            break;
         }
         // Copy xn to prevXn
         foreach(i; 0..n) {
            prevXn[i] = xn[i];
         }
      }
      return xn;
   }
}

/++
Stochastic Gradient descent algorithm.
Reference: Pytorch Stochastic gradient descent
https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html
+/
class stochasticGradientDescent(T) : gradientDescentAlgorithm!T {
private:
   size_t n;
   // Learning rate
   T lr;
   // Momentum
   T mu;
   // Dampening
   T tau;
   // Weight decay
   T decay;
   // Epsilon to check for stopping
   T eps;
   // Nesterov
   bool nesterov;
   // Maximum number of iterations
   size_t iterMax;

public:
   this(size_t size, T learningRate = 1e-3, T momentum = 0, T dampening,
      T weightDecay = 0, T epsilon = 1e-3, bool nesterov = false, size_t nIterMax) {
      n = size;
      lr = learningRate;
      mu = momentum;
      tau = dampening;
      decay = weightDecay;
      eps = epsilon;
      this.nesterov = nesterov;
      iterMax = nIterMax;

   }

   // Optimize a function using stochastic gradient descent
   Slice!(T*) optimize(Slice!(T*) delegate(Slice!(const(T)*)) gradF) {
      auto rng = Random(unpredictableSeed);
      auto prevXn = randomSlice(rng, NormalVariable!T(), n);
      auto xn = empty!T(n);
      auto diff = empty!T(n);
      auto b = empty!T(n);
      auto prevB = empty!T(n);
      foreach(iter; 0..iterMax) {
         // Compute the gradient
         Slice!(T*) grad = gradF(prevXn);
         // weights decay
         if (decay != 0) {
            foreach(i; 0..n) {
               grad[i] = grad[i] + decay * prevXn[i];
            }
         }
         // Momentum
         if (mu != 0) {
            if (iter >0) {
               // set b = mu * prevB + (1 - tau) * grad
               foreach(i; 0..n) {
                  b[i] = mu * prevB[i] + (1 - tau)* grad[i];
               }
            } else {
               // Set b = g.copy
               foreach(i; 0..n) {
                  b[i] = grad[i];
               }
            }
            // Copy b into prevB
            foreach(i; 0..n) {
               prevB[i] = b[i];
            }
            if (nesterov) {
               // set grad = grad + mu * b
               foreach(i; 0..n) {
                  grad[i] = grad[i] + mu * b[i];
               }
            } else {
               // set grad = b
               foreach(i; 0..n) {
                  grad[i] = b[i];
               }
            }
         }
         foreach(i; 0..n) {
            xn[i] = prevXn[i] - lr * grad[i];
         }
         // Check for stopping
         foreach(i; 0..n) {
            diff[i] = xn[i] - prevXn[i];
         }
         T norm = nrm2!T(diff);
         if (norm <= eps) {
            break;
         }
         // Copy xn to prevXn
         foreach(i; 0..n) {
            prevXn[i] = xn[i];
         }
      }
      return xn;
   }
}

