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
      foreach(iter; 0..iterMax) {
         auto grad = gradF(prevXn);
         foreach(i; 0..n) {
            xn[i] = prevXn[i] - lr * grad[i];
         }
         // Check for stopping
         Slice!(T*) diff = empty!T(n);
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


