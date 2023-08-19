 function main()
  s = 5;
    
  n = rand(s,1);



  i = 0;
  while (i < 100)
    W = updateW(s);
    n =sigmoid(resize(W*n)); 
    i++;
    decide(n(1))
    
  end
  
  
  
end


function newW = updateW(s) % This function completely randomizes weights
  W = (2 * rand(s) - 1);
  newW = W .* (eye(s) - ones(s)); % This is mistake which has no effect on overall program, it accidentally multiplies all values by -1
end

function newN = resize(n)
  resize_factor = .5;
  newN = n * resize_factor * size(n,1) / sum (n);
end

function res = decide(x)
  if x >= .5 
    res = true;
  else
    res = false;
  endif
end