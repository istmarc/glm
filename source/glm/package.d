module glm;

import glm.optim;

import mir.ndslice;
import numir;
import mir.blas;
import mir.lapack;

/++
Backward subtitution
Solve Ux = y where U is an upper traingular matrix
+/
void backward_subt(T)(Slice!(const(T)*, 2) U, Slice!(const(T)*) y, Slice!(T*) x) {
   long n = cast(long) x.elementCount;
   assert(y.elementCount == n);
   assert(U.shape[0] == n && U.shape[1] == n);
   x[n-1] = y[n-1] / U[n-1, n-1];
   for (long i = n-2; i>=0; i--) {
      T s = y[i];
      for (long j = i+1; j<n; j++) {
         s -= U[i, j]* x[j];
      }
      x[i] = s / U[i, i];
   }
}

/++
Forward subtitution
Solve Lx = y where L is a lower triangular matrix
+/
void forward_subt(T)(Slice!(const(T)*, 2) L, Slice!(const(T)*) y, Slice!(T*) x) {
   size_t n = x.elementCount;
   assert(y.elementCount == n);
   assert(L.shape[0] == n && L.shape[1] == n);
   x[0] = y[0] / L[0, 0];
   for (size_t i = 1; i < n; i++) {
      T s = y[i];
      for (size_t j = 0; j < i; j++) {
      s -= L[i, j] * x[j];
      }
      x[i] = s / L[i, i];
   }
}

/++ Method for fitting a linear model
   qr: QR decomposition
   cholesky: Cholesky decomposition
   gd: Gradient descent
+/
enum fitMethod{
   qr,
   cholesky,
   gd
}

/++
Linear model.
+/
class linearModel(T) {
private:
   Slice!(T*) beta;
   bool fitted = false;

public:
   // Constructor
   this() {}

   // Destructor
   ~this() {}

   // Get the coefficients
   Slice!(const(T)*) coef() const { return beta;}

   /// Fit a linear model where `x` is matrix and `y` is a vector
   /// By default using qr decomposition
   void fit(Slice!(const(T)*, 2) x, Slice!(const(T)*) y, fitMethod method = fitMethod.qr, T learningRate = 1e-3, T eps = 1e-3, size_t iterMax = 10000) {
      // x is n*k
      size_t n = x.shape[0];
      size_t k = x.shape[1];
      assert(y.shape[0] == n);

      // X is n*(k+1)
      auto X = empty!T(n, k+1);
      X[0..n, 0] = cast(T)(1);
      X[0..n, 1..k+1] = x;

      // Normal equation
      // XtX is (k+1)*(k+1)
      auto XtX = empty!T(k+1, k+1);
      auto Xt = X.transposed();
      gemm!T(cast(T)(1), Xt, X, cast(T)(0), XtX);

      // Xty is (k+1)
      auto Xty = empty!T(k+1);
      gemv!T(cast(T)(1), Xt, y, cast(T)(0), Xty);

      if (method == fitMethod.qr) {
         fitLinearModelQR(XtX, Xty);
      } else if (method == fitMethod.cholesky) {
         fitLinearModelCholesky(XtX, Xty);
      } else if (method == fitMethod.gd) {
         fitLinearModelGradientDescent(XtX, Xty, learningRate, eps, iterMax);
      }
   }

   /// Fit a linear model wehre `x` and `y` are both vectors
   /// by default using qr decomposition
   void fit(Slice!(const(T)*) x, Slice!(const(T)*) y, fitMethod method = fitMethod.qr, T learningRate = 1e-3, T eps = 1e-3, size_t iterMax = 10000) {
      size_t n = y.shape[0];
      assert(n == x.shape[0]);
      size_t k = 1;
      // x is n*k = n * 1
      // X is n*(k+1)
      auto X = empty!T(n, k+1);
      X[0..n, 0] = cast(T)(1);
      X[0..n, 1] = x;

      // Normal equation
      // XtX is (k+1)*(k+1)
      auto XtX = empty!T(k+1, k+1);
      auto Xt = X.transposed();
      gemm!T(cast(T)(1), Xt, X, cast(T)(0), XtX);

      // Xty is (k+1)
      auto Xty = empty!T(k+1);
      gemv!T(cast(T)(1), Xt, y, cast(T)(0), Xty);

      if (method == fitMethod.qr) {
         fitLinearModelQR(XtX, Xty);
      } else if (method == fitMethod.cholesky) {
         fitLinearModelCholesky(XtX, Xty);
      } else if (method == fitMethod.gd) {
         fitLinearModelGradientDescent(XtX, Xty, learningRate, eps, iterMax);
      }
   }

   Slice!(T*) predict(Slice!(const(T)*) x) {
      assert(fitted, "Model must be fitted before doing predictions");
      size_t n = x.elementCount;
      auto X = empty!T(n, 2);
      X[0..n, 0] = cast(T)(1);
      X[0..n, 1] = x;
      auto y = empty!T(n);
      gemv!T(cast(T)(1), X, beta, cast(T)(0), y);
      return y;
   }

   Slice!(T*) predict(Slice!(const(T)*, 2) x) {
      assert(fitted, "Model must be fitted before doing predictions");
      size_t n = x.shape[0];
      size_t k = x.shape[1];
      assert(k + 1 == beta.elementCount);
      auto X = empty!T(n, k+1);
      X[0..n, 0] = cast(T)(1);
      X[0..n, 1..k+1] = x;
      auto y = empty!T(n);
      gemv!T(cast(T)(1), X, beta, cast(T)(0), y);
      return y;
   }

private:
   // Fit a linear model using QR decomposition
   void fitLinearModelQR(Slice!(const(T)*, 2) XtX, Slice!(const(T)*) Xty) {
      // Canonical slice kind
      auto sXtX = empty!T(XtX.shape).canonical;
      // Copy XtX into sXtX
      for (size_t i = 0; i < XtX.elementCount; i++) {
         sXtX.iterator[i] = XtX.iterator[i];
      }
      size_t n = XtX.shape[0];
      assert(XtX.shape[1] == n);
      auto tau = empty!T([n]).canonical;
      auto work = empty!T([n]).canonical;
      auto info1 = geqrf(sXtX, tau, work);
      assert(info1 == 0);
      // Copy the upper triangular matrix into R
      auto R = zeros!T(n, n);
      foreach(i; 0..n){
         foreach(j; i..n){
            R[i,j] = sXtX[j, i];
         }
      }
      auto info2 = orgqr(sXtX, tau, work);
      assert(info2 == 0);
      alias Qt = sXtX;
      // Solve R*beta = b where b=Qt*Xty
      auto b = empty!T(n);
      gemv!T(cast(T)(1), Qt, Xty, cast(T)(0), b);

      beta = empty!T(n);
      backward_subt(R, b, beta);
      fitted = true;
   }

   // Fit a linear model using Cholesky decomposition
   void fitLinearModelCholesky(Slice!(const(T)*, 2) XtX, Slice!(const(T)*) Xty) {
      // Canonical slice kind
      auto sXtX = empty!T(XtX.shape).canonical;
      // Copy XtX into sXtX
      for (size_t i = 0; i < XtX.elementCount; i++) {
         sXtX.iterator[i] = XtX.iterator[i];
      }
      size_t n = XtX.shape[0];
      assert(XtX.shape[1] == n);
      auto info = potrf!T('L', sXtX);
      assert(info == 0);
      auto L = zeros!T(n, n);
      foreach(i; 0..n) {
         foreach(j; 0..i+1) {
            L[i,j] = sXtX[j, i];
         }
      }
      auto z = empty!T(n);
      forward_subt!T(L, Xty, z);
      beta = empty!T(n);
      auto Lt = zeros!T(n, n);
      foreach(i; 0..n) {
         foreach(j; 0..i+1) {
            Lt[j, i] = L[i, j];
         }
      }
      backward_subt!T(Lt, z, beta);
      fitted = true;
   }

   // Fit a linear model using gradient descent
   void fitLinearModelGradientDescent(Slice!(const(T)*, 2) XtX, Slice!(const(T)*) Xty, T learningRate, T eps, size_t iterMax) {
      size_t n = XtX.shape[0];
      assert(XtX.shape[1] == n);

      // Run gradient descent algoithm
      auto algo = new gradientDescent!T(n, learningRate, eps, iterMax);
      // Gradient of the objective function
      Slice!(T*) gradObj(Slice!(const(T)*) theta) {
         size_t n = theta.elementCount;
         auto XtXtheta = empty!T(n);
         gemv!T(cast(T)(1), XtX, theta, cast(T)(0), XtXtheta);
         auto gradValue = empty!T(n);
         foreach(i; 0..n) {
            gradValue[i] = cast(T)(-2)*Xty[i] + cast(T)(2)*XtXtheta[i];
         }
         return gradValue;
      }
      // Optimize the objective function
      beta = algo.optimize(&gradObj);

      fitted = true;
   }
}

