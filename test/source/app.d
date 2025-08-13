import glm;

import std.stdio;
import std.array;

import mir.ndslice;
import numir;

/*
import ggplotd.aes: aes, Aes, merge;
import ggplotd.axes: xaxisLabel, yaxisLabel;
import ggplotd.ggplotd: GGPlotD, putIn, title;
import ggplotd.geom: geomPoint, geomBox, geomLine;
*/

void main()
{
   auto x = [1.47f, 1.50f, 1.52f, 1.55f, 1.57f, 1.60f, 1.63f, 1.65f, 1.68f, 1.70f, 1.73f, 1.75f, 1.78f, 1.80f, 1.83f].sliced(15);
   auto y = [52.21f, 53.12f, 54.48f, 55.84f, 57.20f, 58.57f, 59.93f, 61.29f, 63.11f, 64.47f, 66.28f, 68.10f, 69.92f, 72.19f, 74.46f].sliced(15);

   {
      writeln("y~beta_0 + beta_1 x using QR");
      auto xs = x.array;
      auto ys = y.array;


      auto model = new linearModel!float();
      model.fit(x, y);
      auto coef = model.coef();
      writeln("Fitted coef = ", coef);

      auto pred = model.predict(x);
      writeln("true y    = ", y);
      writeln("predicted = ", pred);
      /*
      // TODO Plot the solution
      // TODO Plot in the same figure
      auto a = Aes!(float[], "x", float[], "y")(xs, ys);
      auto gg1 = geomPoint(a).putIn(GGPlotD());
      gg1.save("scatter.png");
      auto b = Aes!(float[], "x", float[], "y")(xs, pred.array);
      auto gg2 = geomLine(b).putIn(GGPlotD());
      gg2.save("line.png");
      */
   }

   {
      writeln("y~beta_0 + beta_1 x + beta_2 x^2 using QR");
      auto x2 = x*x;
      size_t n = x.elementCount;
      auto X = empty!float([n, 2uL]);
      X[0..n, 0] = x;
      X[0..n, 1] = x2;

      auto model = new linearModel!float();
      model.fit(X, y);
      auto coef = model.coef();
      writeln("Fitted coef = ", coef);
      auto pred = model.predict(X);
      writeln("true y    = ", y);
      writeln("predicted = ", pred);
   }

   {
      writeln("y~beta_0 + beta_1 x + beta_2 x^2 using cholesky");
      auto x2 = x*x;
      size_t n = x.elementCount;
      auto X = empty!float([n, 2uL]);
      X[0..n, 0] = x;
      X[0..n, 1] = x2;

      auto model = new linearModel!float();
      model.fit(X, y, fitMethod.cholesky);
      auto coef = model.coef();
      writeln("Fitted coef = ", coef);
      writeln(X.shape);
      auto pred = model.predict(X);
      writeln("true y    = ", y);
      writeln("predicted = ", pred);
   }

}
