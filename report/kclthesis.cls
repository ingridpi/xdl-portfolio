\ProvidesClass{kclthesis}
\NeedsTeXFormat{LaTeX2e}

\newif\ifkclthesis@releaseproject
\newif\ifkclthesis@notreleaseproject
\newif\ifkclthesis@undreleaseproject

\DeclareOption{releaseproject}{\kclthesis@releaseprojecttrue\kclthesis@notreleaseprojectfalse\kclthesis@undreleaseprojectfalse}
\DeclareOption{notreleaseproject}{\kclthesis@releaseprojectfalse\kclthesis@notreleaseprojecttrue\kclthesis@undreleaseprojectfalse}
\DeclareOption{undreleaseproject}{\kclthesis@releaseprojectfalse\kclthesis@notreleaseprojectfalse\kclthesis@undreleaseprojecttrue}
\ExecuteOptions{undreleaseproject}

\ProcessOptions*% process options in the order the user provides them

\LoadClass[11pt, a4paper]{report}

\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage{amsfonts}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{bm}
\RequirePackage{booktabs}
\RequirePackage{color}
\RequirePackage{fancyhdr}
\RequirePackage{float}
\RequirePackage{framed}
\RequirePackage{geometry}[paper=a4paper]
\RequirePackage{glossaries}[acronym,nonumberlist]
\RequirePackage{graphbox}
\RequirePackage{graphicx}
\RequirePackage{hyperref}
\RequirePackage{inputenc}[utf8]
\RequirePackage{longtable}
\RequirePackage{minted}
\RequirePackage{multirow}
\RequirePackage{nomencl}
\RequirePackage{pgfgantt}
\RequirePackage{setspace}
\RequirePackage{subfigure}
\RequirePackage{todonotes}[textsize=tiny,figwidth=8cm]
\RequirePackage{tocbibind}[nottoc,notlot,notlof]
\RequirePackage{wasysym}

%% Nomenclature
\renewcommand\nomgroup[1]{%
  \item[\bfseries
    \ifstrequal{#1}{A}{Numbers and Arrays}{%
      \ifstrequal{#1}{B}{Sets}{%
        \ifstrequal{#1}{C}{Indexing}{%
          \ifstrequal{#1}{D}{Algebra Operations}{%
            \ifstrequal{#1}{E}{Probability}{%
              \ifstrequal{#1}{F}{Functions}{%
                \ifstrequal{#1}{G}{Markov Decision Process}{%
                  \ifstrequal{#1}{Z}{Other Symbols}{}
                }
              }%
            }%
          }%
        }%
      }%
    }%
  ]
}

% Macros
\newcommand*{\modulecode}[1]{\gdef\@modulecode{#1}}
\newcommand*{\submissiontitle}[1]{\gdef\@submissiontitle{#1}}
\newcommand*{\studentnumber}[1]{\gdef\@studentnumber{#1}}
\newcommand*{\programme}[1]{\gdef\@programme{#1}}
\newcommand*{\supervisor}[1]{\gdef\@supervisor{#1}}
\newcommand*{\wordcount}[1]{\gdef\@wordcount{#1}}
\newcommand*{\signdate}[1]{\gdef\@signdate{#1}}
\newcommand*{\@signdate}{\today}
\newcommand*{\ReleaseProject}[1]{\gdef\@ReleaseProject{#1}}
\newcommand*{\department}[1]{\gdef\@department{#1}}
% \newcommand*{\version}[1]{\gdef\@version{#1}}

% New Comment for todonotes
\newcommand{\insertref}[1]{\todo[color=green!40]{#1}}
\newcommand{\explainindetail}[1]{\todo[color=red!40]{#1}}
\newcommand{\askfeedback}[1]{\todo[color=blue!40]{#1}}

\renewcommand*{\maketitle}{

\begin{titlepage}
\thispagestyle{empty}
\null\vskip0.2in%

\begin{figure}[H]
\centering
\includegraphics[width=5cm]{figures/kcl.png}
\end{figure}

\begin{center}
\Large Department of \@department \\ 
King's College London\\ 
United Kingdom
\\~\\
\@modulecode ~- Individual Project
\end{center}

\vspace{1cm}

\begin{center}
\LARGE{{\bf 
\@title
}}\\
\vspace{0.2cm}
\vskip\bigskipamount
\leaders\vrule width \textwidth\vskip0.4pt 
\vskip\bigskipamount % ditto
\nointerlineskip
\end{center}

\vspace{0.2cm}

\begin{center}
\doublespacing
{Name: \bf \@author}\\
Student Number: \@studentnumber\\
Course: \@programme \\ 
{Supervisor: \bf \@supervisor} \\
{Word count: \@wordcount} \\
\end{center}

\vspace{1.2cm}

\end{titlepage}
}

\endinput
