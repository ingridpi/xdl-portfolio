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

% Mathfonts
\RequirePackage{amsmath, amsfonts, amsthm, amssymb}
\RequirePackage{wasysym}

%Additional Packages
\RequirePackage{color}
\RequirePackage[nottoc,notlot,notlof]{tocbibind}
\RequirePackage{hyperref}
\RequirePackage{fancyhdr}
\RequirePackage{subfigure}
\RequirePackage[textsize=tiny,figwidth=8cm]{todonotes}

% Cover Page
\RequirePackage{graphicx}
\RequirePackage[paper=a4paper]{geometry}
\RequirePackage{setspace}
\RequirePackage{framed}


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

% Second Cover Page

\begin{titlepage}
\thispagestyle{empty}
\null\vskip0.2in%

\begin{figure}[H]
\centering
\includegraphics[width=3.25cm]{figures/kcl.png}
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

% \begin{center}
%     \@version
%     \\
%     \vspace{0.2cm}
%     \vskip\bigskipamount
%     \nointerlineskip
% \end{center}

\vspace{0.2cm}

\begin{center}
\doublespacing
{Name: \bf \@author}\\
Student Number: \@studentnumber\\
Course: \@programme \\ 
Supervisor: \bf \@supervisor
\end{center}

\vspace{1.2cm}

\end{titlepage}
}

\endinput
