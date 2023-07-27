/+  *lagoon
:-  %say
|=  *
:-  %noun
|^
=,  la
=/  a   (en-ray [~[2 2] 5 %float] ~[~[.2.5409 .-0.0113] ~[.-0.0113 .0.5287]])
=/  b   (en-ray [~[2 1] 5 %float] ~[~[.1.3864] ~[.0.3719]])
=/  x0  (en-ray [~[2 1] 5 %float] ~[~[.1] ~[.1]])
~>  %bout
(conjgrad a b x0)
::  conjugate gradient descent method
::    a  is a 2D system of linear equations
::    b  is the corresponding solutions
::    x0 is a starting guess at the unknowns
::
++  conjgrad
  =,  la
  |=  [a=ray b=ray x0=ray]
  ^-  ray
  =/  r=ray  (sub b (matmul-2d a x0))
  =/  p=ray  r
  =/  rsold=ray  (dot r r)
  ::
  =/  i=@ud  0
  =/  x=ray  x0
  |-  ^-  ray
  ~&  >  i
  ~&  >>  x
  ~&  >>>  rsold
  ?:  =(i &1.shape.meta.b)  x
  =/  ap=ray  (matmul-2d a p)
  =/  alpha=ray  (div rsold (dot p ap))
  =/  r  (sub r (mul (fill meta.ap data.alpha) ap))
  =/  rsnew=ray  (dot r r)
  ?:  (all (lth rsnew (scalarize meta.rsnew %rs .1e-5)))  x
  %=  $
    i      +(i)
    p      (add r (mul (fill meta.p data:(div rsnew rsold)) p))
    rsold  rsnew
    x      (add x (mul (fill meta.p data.alpha) p))
  ==
--
