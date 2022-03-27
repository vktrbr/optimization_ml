.. raw:: latex

    \begin{center}
        \begin{tikzpicture}[font=\small,thick]

      % Making nodes
        \node[draw,
        rounded rectangle,
        minimum width=2.5cm,
        minimum height=1cm] (start) {start};

        \node[draw,
        trapezium,
        trapezium left angle = 65,
        trapezium right angle = 115,
        trapezium stretches,
        below=of start,
        minimum width=3.5cm,
        minimum height=1cm
        ] (input) {Given $\mathrm{x}_0$, $\varepsilon$, $k_{\max}$ $\quad k = 0$};

        \node[draw,
        below=of input,
        minimum width=3.5cm,
        minimum height=1cm
        ] (p0) {Set $ \ p_k = - \nabla f(\mathrm{x}_k)$};

        \node[draw,
        below=of p0,
        minimum width=3.5cm,
        minimum height=1cm
        ] (grad) {Evaluate $ \ \nabla f(\mathrm{x}_k)$};

        \node[draw,
        diamond,
        below=1cm of grad,
        minimum width=4.5cm,
        inner sep=0] (condition) {\shortstack{$ \| \nabla f(\mathrm{x}_{k}) \|_2 < \varepsilon$ \\ $\mathbf{or} \ k \geq k_{\max} \quad$ }};


        \node[draw,
        right=1.2cm of condition,
        minimum width=2.5cm,
        minimum height=1cm,] (ls_gamma) {linesearch of $\gamma$ };

        \node[draw,
        diamond,
        right=of ls_gamma,
        minimum width=2.5cm,
        minimum height=1cm,] (ls_converged) {\shortstack{linesearch of \\ $\gamma$ converged}};


        \node[draw,
        right=1.2cm of ls_converged,
        minimum width=2.5cm,
        minimum height=1cm,] (brent_gamma) {find $\gamma$ by Brent};

        \node[draw,
        above=3cm of ls_converged,
        minimum width=2.5cm,
        minimum height=1cm,] (x_new) {$\mathrm{x}_{k+1} = \mathrm{x}_k + \gamma \cdot p_k$};

        \node[draw,
        above=of x_new,
        minimum width=2.5cm,
        minimum height=1cm,] (beta) {Evaluate $\ \beta^{FR}_{k+1}$};

        \node[draw,
        right=1.2cm of grad,
        minimum width=2.5cm,
        minimum height=1cm,] (pk) {\shortstack{$p_{k+1} = \nabla f_{k+1} + \beta^{FR}_{k+1} p_k$ \\ $k = k + 1$}};

        \node[draw,
        rounded rectangle,
        below=2cm of condition,
        minimum width=2.5cm,
        minimum height=1cm,] (end) { return $\mathrm{x}_{k}$};

     %making edges
      \draw[-latex] (start) edge (input);

      %making edges
      \draw[-latex] (start) edge (input)
                      (input) edge (p0)
                      (p0) edge (grad)
                      (grad) edge (condition)
                      (ls_gamma) edge (ls_converged)
                      (x_new) edge (beta)
                      (beta) -| (pk)
                      (pk) edge (grad);


      \draw[-stealth] (brent_gamma) |- (x_new);

      \draw[-stealth] (condition) -- (ls_gamma)
      node[pos=0.5, anchor=south]{No};

      \draw[-stealth] (condition) -- (end)
      node[pos=0.5,fill=white,inner sep=5]{Yes};

      \draw[-stealth] (ls_converged) -- (brent_gamma)
      node[pos=0.5, anchor=south]{No};

      \draw[-stealth] (ls_converged) -- (x_new)
      node[pos=0.5,fill=white,inner sep=5]{Yes};

    \end{tikzpicture}
    \end{center}