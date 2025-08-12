module glm;

import mir.ndslice;
import numir;
import mir.blas;
import mir.lapack;

/++
Backward subtitution
Solve Ux = y
+/
void backward_subt(T)(Slice!(const(T)*, 2) U, Slice!(const(T)*) y, Slice!(T*) x) {
	long n = cast(long) x.shape[0];
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

/++ Method for fitting a linear model
	qr: QR decomposition
	cholesky: Cholesky decomposition
	svd: Singular value decomposition
+/
enum fitMethod{
	qr,
	cholesky,
	svd
}

/++
Linear model.
+/
class linearModel(T) {
private:
	Slice!(T*) beta;

public:
	// Constructor
	this() {}

	// Destructor
	~this() {}

	// Get the coefficients
	Slice!(const(T)*) coef() const { return beta;}

	/// Fit a linear model where `x` is matrix and `y` is a vector
	/// By default using qr decomposition
	void fit(Slice!(const(T)*, 2) x, Slice!(const(T)*) y, fitMethod method = fitMethod.qr) {
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
		}
	}

	/// Fit a linear model wehre `x` and `y` are both vectors
	/// by default using qr decomposition
	void fit(Slice!(const(T)*) x, Slice!(const(T)*) y, fitMethod method = fitMethod.qr) {
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
		}
	}

	Slice!(T*) predict(Slice!(const(T)*) x) {
		size_t n = x.elementCount;
		auto X = empty!T(n, 2);
		X[0..n, 0] = cast(T)(1);
		X[0..n, 1] = x;
		auto y = empty!T(n);
		gemv!T(cast(T)(1), X, beta, cast(T)(0), y);
		return y;
	}

	Slice!(T*) predict(Slice!(const(T)*, 2) x) {
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
	}
}

