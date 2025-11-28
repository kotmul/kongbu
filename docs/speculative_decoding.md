```latex
\begin{algorithm}[t]
    \small
    \centering
    \caption{Speculative knowledge distillation}\label{alg}
    \begin{algorithmic}[1]
    \Require{Student LLM $M_s$, Teacher LLM $M_t$, Prompt dataset $\{x_j\}_{j=1}^N$, Decoding length $\alpha$}.\\
    \textbf{Hyperparameters}: Top $K$ for token acceptance, Divergence metric $D$,  
    
    \For{$step := 1$ to $N$} \Comment{We assume batch size $:= 1$ to simplify the illustration}
        % \State $Loss := 0$
        \For{$i := 1$ to $\alpha$} 
            % \State $q_i(y) \leftarrow M_s(x_j + [y_1, y_2, ..., y_{i-1}])$
            
            \State $y_{i} \sim M_s(.|y_{<i},x_j)$ \Comment{Sample $y_{i}$ from student $M_s$}

            \If {$y_{i} \not\subset top_K(M_t(.|y_{<i},x_j))$} \Comment{If $y_{i}$ is not within top $K$ token of teacher $M_t$} %\ElsIf 
            
            % \State $Loss$ += $D(M_t(.|y_{<i},x_j)||M_s(.|y_{<i},x_j))$
            % \Else 
            
            \State $y_{i} \sim M_t(.|y_{<i},x_j)$ \Comment{We replace student token to teacher's resampled token}
            \EndIf   
            
            % \State $Loss$ += $D(M_t(.|y_{<i},x_j)||M_s(.|y_{<i},x_j))$
              

            % \If {$y_{i}:=[EOS]$}
            %     \State End the decoding loop
            % \EndIf 
    \EndFor
    \State Apply gradient descent to minimize the minibatch loss  $D(M_t||M_s)(y|x)$ in \eqref{eqn:distance_metric}
    \EndFor
    % \Ensure Student LLM $M_{s_N}$
    \end{algorithmic}
\label{algo:specKD}
\end{algorithm}
```