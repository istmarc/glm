module glm.linalg;

import mir.ndslice;

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

