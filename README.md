# glm
Generalized linear models in D

## Example usage

- Fitting a linear model using QR decomposition

```d
import glm;

void main()
{
    auto x = [1.47f, 1.50f, 1.52f, 1.55f, 1.57f, 1.60f,
              1.63f, 1.65f, 1.68f, 1.70f, 1.73f, 1.75f,
              1.78f, 1.80f, 1.83f].sliced(15);
    auto y = [52.21f, 53.12f, 54.48f, 55.84f, 57.20f,
              58.57f, 59.93f, 61.29f, 63.11f, 64.47f,
              66.28f, 68.10f, 69.92f, 72.19f, 74.46f].sliced(15);

    auto model = new LinearModel!float();
    model.fit(x, y);

    auto coef = model.coef();
    writeln("Fitted coef = ", coef);

    auto pred = model.predict(x);
    writeln("true y    = ", y);
    writeln("predicted = ", pred);
}
```
- Fit using a different method

```d
    model.fit(x, y, FitMethod.cholesky);
```

`FitMethod` can be `FitMethod.qr` (default), `FitMethod.cholesky`, `FitMethod.gd` (gradient descent), or `FitMethod.sgd` (Stochastic gradient descent).


