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
          ] (input) {Given $\mathrm{x}_0$, $\varepsilon$, $k_{\max}$, $\quad k = 0$};


          \node[draw,
            below=of input,
            minimum width=3.5cm,
            minimum height=1cm
          ] (grad) {Evaluate $\nabla f(\mathrm{x}_k)$};

          \node[draw,
            diamond,
            below=1cm of grad,
            minimum width=4.5cm,
            inner sep=0] (condition) {\shortstack{$ \| \nabla f(\mathrm{x}_{k}) \|_2 < \varepsilon$ \\ $\mathbf{or} \ k \geq k_{\max} \quad$ }};

          \node[draw,
            right=3.5cm of condition,
            minimum width=4cm,
            minimum height=1cm
          ] (no_condition_1) {$\displaystyle \gamma = \operatorname*{arg\,min}_{\gamma \in (0, 1)}\phi(\gamma)$};

          \node[draw,
            above=of no_condition_1,
            minimum width=4cm,
            minimum height=1cm
          ] (x_new) {\shortstack{$\mathrm{x}_{k+1} = \mathrm{x}_k - \gamma \cdot \nabla f(\mathrm{x}_k) $ \\ $k = k + 1$}};

          \node[draw,
            rounded rectangle,
            below=2cm of condition,
            minimum width=2.5cm,
            minimum height=1cm,] (end) { return $\mathrm{x}_{k}$};

          %making edges
          \draw[-latex] (start) edge (input);

          %making edges
          \draw[-latex] (start) edge (input)
                          (input) edge (grad)
                          (grad) edge (condition)
                          (no_condition_1) edge (x_new)
                          (x_new) |- (grad);

          \draw[-stealth] (condition) -- (no_condition_1)
          node[pos=0.5,fill=white,inner sep=5]{No};

          \draw[-stealth] (condition) -- (end)
          node[pos=0.5,fill=white,inner sep=5]{Yes};


        \end{tikzpicture}
    \end{center}