filetype on
filetype indent on
syntax on

set number			" show line number
set wrap			" wrap long line

set cursorline 			" hightlight current line
hi CursorLine cterm=NONE ctermbg=darkgreen ctermfg=white

set showcmd
set wildmenu			" visual autocomplete for command menu

" search setting
set hlsearch			" highlight matched word
set smartcase
set ignorecase			" insensitive searching
set incsearch
set showmatch			" show matching braces

" tab and indentation setting
set autoindent			" auto-indent new lines
set shiftwidth=4		" number of auto-indent spaces
set smartindent
set smarttab
set expandtab
set softtabstop=4		" use 4 spaces instead of tab

set undolevels=1000
set backspace=indent,eol,start

let g:netrw_banner = 0
let g:netrw_liststyle = 3
let g:netrw_browse_split = 4
let g:netrw_altv = 1
let g:netrw_winsize = 25
augroup ProjectDrawer
  autocmd!
  autocmd VimEnter * :Vexplore
augroup END
