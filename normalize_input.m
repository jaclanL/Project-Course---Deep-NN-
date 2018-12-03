function normInputSequence = normalize_input(inputSequence)
iMu = mean(inputSequence, 1);
[iMax,~] = max(abs(inputSequence),[], 1);
normInputSequence = (inputSequence-iMu)./iMax;
end

