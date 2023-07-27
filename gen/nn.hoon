/+  *lagoon, nn=layer
:-  %say
|=  *
:-  %noun
=,  la
|^
::  build a neural network
=/  =meta  [~[4 2] 5 %float]
=/  train=ray
  %+  en-ray
    [~[1 2] 5 %float]
  :~  ::~[.0 .0]
      ::~[.1 .0]
      ::~[.0 .1]
      ~[.1 .1]
  ==
::
=/  hwts=ray   (en-ray [~[2 2] 5 %float] ~[~[.3.6721 .3.6709] ~[.5.87448 .5.86795]])
=/  hbias=ray  (en-ray [~[1 2] 5 %float] ~[~[.-5.61072 .-2.4229]])
=/  fwts=ray   (en-ray [~[2 1] 5 %float] ~[~[.-8.0687] ~[.7.45091]])
=/  fbias=ray  (fill [~[1 1] 5 %float] data:(scalarize meta %rs .-3.35558))
::
=/  hout
  %-  sigmoid:nn
    ::  first hidden layer
    %+  add
      %+  matmul-2d
        train
      hwts
    hbias
::
=/  out
  %-  sigmoid:nn
    ::  second hidden layer
    %+  add
      %+  matmul-2d
        hout
      fwts
    fbias
`(list @rs)`(ravel out)
::  conjugate gradient descent method
::    a  is a 2D system of linear equations
::    b  is the corresponding solutions
::    x0 is a starting guess at the unknowns
::
++  conjgrad
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
