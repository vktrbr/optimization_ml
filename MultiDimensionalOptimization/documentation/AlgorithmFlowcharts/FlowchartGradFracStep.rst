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
          ] (input) {Given $\mathrm{x}_0$, $\varepsilon$, $\gamma$, $\delta$, $\lambda$, $k_{\max}$, $\quad k = 0$};


          \node[draw,
            below=of input,
            minimum width=3.5cm,
            minimum height=1cm
          ] (cycle_start) {$t = \mathrm{x}_k - \gamma \cdot \nabla f(\mathrm{x}_k)$};

          \node[draw,
            diamond,
            below=1cm of cycle_start,
            minimum width=4.5cm,
            inner sep=0] (condition) {\shortstack{$ f(t) - f(\mathrm{x}_k) \leq $ \\ $\leq - \gamma \cdot \delta \cdot \|\nabla f(\mathrm{x}_k) \|^2_2$ }};

          \node[draw,
            right=2cm of condition,
            minimum width=3.5cm,
            minimum height=1cm
          ] (no_condition_1) {$\gamma = \gamma \cdot \lambda$};

          \node[draw,
            below = 2cm of condition,
            minimum width=4cm,
            minimum height=1cm
          ] (yes_condition_1) {$\mathrm{x}_{k+1} = \mathrm{x}_k - \gamma \cdot \nabla f(\mathrm{x}_k) $};

          \node[draw,
            diamond,
            below=1cm of yes_condition_1,
            minimum width=3.5cm,
            inner sep=0] (stop_criteria) { \shortstack{$ \| \nabla f(\mathrm{x}_{k+1}) \|_2 < \varepsilon$ \\ $ \mathbf{or} \ k + 1 \geq k_{\max} \quad $ }};

          \node[draw,
            left = 2cm of stop_criteria,
            minimum width=3.5cm,
            minimum height=1cm
          ] (no_condition_2) {$k = k + 1$};


          \node[draw,
            rounded rectangle,
            below=2cm of stop_criteria,
            minimum width=2.5cm,
            minimum height=1cm,] (end) { return $\mathrm{x}_{k+1}$};

          %making edges
          \draw[-latex] (start) edge (input)
          (input) edge (cycle_start)
          (cycle_start) edge (condition)
          (yes_condition_1) edge (stop_criteria)
          (no_condition_2) |- (cycle_start);

          \draw[-stealth] (no_condition_1) |- (cycle_start);

          \draw[-stealth] (condition) -- (no_condition_1)
          node[pos=0.5,fill=white,inner sep=5]{No};

          \draw[-stealth] (condition) -- (yes_condition_1)
          node[pos=0.5,fill=white,inner sep=5]{Yes};

          \draw[-stealth] (stop_criteria) -- (no_condition_2)
          node[pos=0.5,fill=white,inner sep=5]{No};

          \draw[-stealth] (stop_criteria) -- (end)
          node[pos=0.5,fill=white,inner sep=5]{Yes};


        \end{tikzpicture}

    \end{center}