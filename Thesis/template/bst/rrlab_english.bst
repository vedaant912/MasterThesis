% BibTeX `unika-bib' style file for BibTeX version 0.99c, LaTeX version 2.09
% Place it in a file called unika.bst in the BibTeX search path. 
% (Placing it in the same directory as the LaTeX document should also work.)

% This version was made by modifying the master file made by
% Oren Patashnik (PATASHNIK@SCORE.STANFORD.EDU)

% Copyright (C) 1985, all rights reserved.
% Modifications Copyright 1991, W. Gerteis
% Copying of this file is authorized only if either
% (1) you make absolutely no changes to your copy, including name, or
% (2) if you do make changes, you name it something other than
% btxbst.doc, plain.bst, unsrt.bst, alpha.bst, and abbrv.bst.
% This restriction helps ensure that all standard styles are identical.

% There are undoubtably bugs in this style.  If you make bug fixes,
% improvements, etc.  please let me know.  My e-mail address is:
%	gerteis@telematik.informatik.uni-karlsruhe.de

%   Citation format: [author-last-name year]
%
%   Reference list ordering: alphabetical by author or whatever passes
%	for author in the absence of one.
%

ENTRY
  { address
    author
    booktitle
    chapter
    edition
    editor
    howpublished
    institution
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    volume
    year
  }
  {}
  { label extra.label sort.label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}

STRINGS { s t bauthor}

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
	{ add.period$ write$
	  newline$
	  "\newblock " write$
	}
	{ output.state before.all =
	    'write$
	    { add.period$ " " * write$ }
	  if$
	}
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {output.bibitem}
{ newline$

  "\bibitem[" write$
  label write$
  "]{" write$

  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
	'skip$
	{ after.sentence 'output.state := }
      if$
    }
  if$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}

FUNCTION {new.block.checka}
{ empty$
    'skip$
    'new.block
  if$
}

FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}

FUNCTION {new.sentence.checka}
{ empty$
    'skip$
    'new.sentence
  if$
}

FUNCTION {new.sentence.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.sentence
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "{\em " swap$ * "}" * }
  if$
}

FUNCTION {bold}
% top of stack in small caps
{ duplicate$ empty$
    { pop$ "" }
    { "\textbf{" swap$ * "}" * }
  if$
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Determine the length of a string
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
INTEGERS{ l }
FUNCTION{ string.length }
{
  #1 'l :=
  {duplicate$ duplicate$ #1 l substring$ = not} 
    {l #1 + 'l :=}
    while$
    pop$ l
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find and replace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
STRINGS{replace find text}
INTEGERS{find_length}
FUNCTION{find.replace}
{ 'replace := 
  'find :=
  'text :=
  find string.length 'find_length :=
  ""
    { text empty$ not }
    { text #1 find_length substring$ find =
      {
        replace *
        text #1 find_length + global.max$ substring$ 'text :=
      }
      {
        text #1 #1 substring$ *
        text #2 global.max$ substring$ 'text :=
      }
    if$
    }
  while$
}

INTEGERS { nameptr namesleft numnames }

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }

    { s nameptr "{f.~}{vv~}{ll}{, jj}" format.name$ 't :=

      t bauthor =
       { t bold 't := }
       'skip$
      if$

      nameptr #1 >
    { namesleft #1 >
        { ", " * t *}
        { numnames #1 >
        { "" * }
        'skip$
          if$
          t "others" =
        { " et al" * }
        { ", " * t * }
          if$
        }
      if$
    }
    't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$

}

FUNCTION {format.authors}
{ author empty$
    { "" }
    { author "." " " find.replace format.names}
  if$
}

FUNCTION {format.editors}
{ editor empty$
    { "" }
    { editor "." " " find.replace format.names
      editor num.names$ #1 >
	{ ", Eds." * }
	{ ", Ed." * }
      if$
    }
  if$
}

FUNCTION {format.title}
{ title empty$
    { "" }
    
    {"``" title * "''" * }

  if$
}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
	{ t #1 #2 substring$ "--" = not
	    { "--" *
	      t #2 global.max$ substring$ 't :=
	    }
	    {   { t #1 #1 substring$ "-" = }
		{ "-" *
		  t #2 global.max$ substring$ 't :=
		}
	      while$
	    }
	  if$
	}
	{ t #1 #1 substring$ *
	  t #2 global.max$ substring$ 't :=
	}
      if$
    }
  while$
}

FUNCTION {format.date}
{ year empty$
    { month empty$
	{ "" }
	{ "there's a month but no year in " cite$ * warning$
	  month
	}
      if$
    }
    { month empty$
	'year
	{ month n.dashify " " * year * }
      if$
    }
  if$
}

FUNCTION {format.btitle}
{ title emphasize
}

FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}

FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}

FUNCTION {format.series}
{series empty$
 { " " }
 { "ser. " series *} 
 if$
}

FUNCTION {format.volume}
{volume empty$
 { " " }
 { series empty$
   {"vol. " volume *}
   {"vol. " volume *}  
 if$
 }
 if$
}

FUNCTION {format.number}
{number empty$
 { " " }
 { "no. " number n.dashify *} 
 if$
}

FUNCTION {format.edition}
{ edition empty$
    { "" }
	{ edition " ed." * }
      if$
}

INTEGERS { multiresult }

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
	{ #1 'multiresult := }
	{ t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}

FUNCTION {format.pages}
{ pages empty$
    { "" }
    { pages multi.page.check
	{ "pp.\" pages n.dashify tie.or.space.connect }
	{ "p.\" pages tie.or.space.connect }
      if$
    }
  if$
}



FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
	{ "ch.\/" }
	{ type "l" change.case$ }
      if$
      chapter tie.or.space.connect
      pages empty$
	'skip$
	{ ", " * format.pages * }
      if$
    }
  if$
}

FUNCTION {format.in.booktitle}
{ booktitle empty$
    { "" }
    { "in " booktitle emphasize * }
  if$
}

FUNCTION {empty.misc.check}
{ author empty$ title empty$ howpublished empty$
  month empty$ year empty$ note empty$
  and and and and and

  key empty$ not and

    { "all relevant fields are empty in " cite$ * warning$ }
    'skip$
  if$
}

FUNCTION {format.thesis.type}
{ type empty$
    'skip$
    { pop$
      type "t" change.case$
    }
  if$
}

FUNCTION {format.tr.number}
{ type empty$
    { "{T}ech. {R}ep." }
    'type
  if$
  number empty$
    { "t" change.case$ }
    { number tie.or.space.connect }
  if$
}

FUNCTION {format.article.crossref}
{ key empty$
    { journal empty$
	{ "need key or journal for " cite$ * " to crossref " * crossref *
	  warning$
	  ""
	}
	{ "In {\em " journal * "\/}" * }
      if$
    }
    { "In " key * }
  if$
  " \cite{" * crossref * "}" *
}

FUNCTION {format.crossref.editor}
{ editor #1 "{vv~}{ll}" format.name$
  editor num.names$ duplicate$
  #2 >
    { pop$ " et~al." * }
    { #2 <
	'skip$
	{ editor #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
	    { " et~al." * }
	    { " and " * editor #2 "{vv~}{ll}" format.name$ * }
	  if$
	}
      if$
    }
  if$
}

FUNCTION {format.book.crossref}
{ volume empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      "In "
    }
    { "Volume" volume tie.or.space.connect
      " of " *
    }
  if$
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
	{ series empty$
	    { "need editor, key, or series for " cite$ * " to crossref " *
	      crossref * warning$
	      "" *
	    }
	    { "{\em " * "series:\" series * "\/}" * }
	  if$
	}
	{ key * }
      if$
    }
    { format.crossref.editor * }
  if$
  " \cite{" * crossref * "}" *
}

FUNCTION {format.incoll.inproc.crossref}
{ editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
	{ booktitle empty$
	    { "need editor, key, or booktitle for " cite$ * " to crossref " *
	      crossref * warning$
	      ""
	    }
	    { "In {\em " booktitle * "\/}" * }
	  if$
	}
	{ "In " key * }
      if$
    }
    { "In " format.crossref.editor * }
  if$
  " \cite{" * crossref * "}" *
}

FUNCTION {format.address.publisher}
{
 address empty$
   { publisher empty$
      { "" }
      {publisher }
     if$
   }
   { publisher empty$ 
      { address }
      {address  ": " * publisher *}
    if$
   }
 if$
}

FUNCTION {BoldAuthor}
{
  "hui " 'bauthor :=
  author #1 "{f.~}{vv~}{ll}{, jj}" format.name$ 'bauthor :=
}

FUNCTION {article}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  crossref missing$
    { journal emphasize "journal" output.check
      format.volume output
      format.number output
      format.pages output
      format.date "year" output.check
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {book}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors output.nonnull
      crossref missing$
	{ "author and editor" editor either.or.check }
	'skip$
      if$
    }
  if$
  format.btitle "title" output.check
  crossref missing$
    { format.edition output
      format.series output
      author empty$
       'skip$
        {format.editors output}
      if$
      format.address.publisher output
      format.date "year" output.check
      format.volume output
      format.number output
    }
    { new.block
      format.book.crossref output.nonnull
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {booklet}
{ output.bibitem
  format.authors output
  format.title "title" output.check
  howpublished address new.block.checkb
  howpublished output
  address output
  format.date output
  new.block
  note output
  fin.entry
}

FUNCTION {inbook}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors output.nonnull
      crossref missing$
	{ "author and editor" editor either.or.check }
	'skip$
      if$
    }
  if$
  format.btitle "title" output.check
  crossref missing$
    { format.edition output
      format.series output
      format.address.publisher output
      format.date "year" output.check
      format.volume output
      format.number output
      format.chapter.pages output
    }
    { format.chapter.pages output.check
      new.block
      format.book.crossref output.nonnull
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  crossref missing$
    { format.in.booktitle "booktitle" output.check
      format.edition output
      format.series output
      format.editors output
      format.address.publisher output
      format.date "year" output.check
      format.volume output
      format.number output
      format.chapter.pages output
    }
    { format.incoll.inproc.crossref output.nonnull
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  crossref missing$
    { format.in.booktitle "booktitle" output.check
      format.series output
      format.editors output
      format.volume output
      format.number output
      organization output
      new.sentence
      format.address.publisher output
      format.date "year" output.check
      format.pages output
    }
    { format.incoll.inproc.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {conference} { inproceedings }

FUNCTION {manual}
{ output.bibitem
  author empty$
    { organization empty$
	'skip$
	{ organization output.nonnull
	  address output
	}
      if$
    }
    { format.authors output.nonnull }
  if$
  new.block
  format.btitle "title" output.check
  author empty$
    { organization empty$
	{ address new.block.checka
	  address output
	}
	'skip$
      if$
    }
    { organization address new.block.checkb
      organization output
      address output
    }
  if$
  new.sentence
  format.edition output
  format.date output
  new.block
  note output
  fin.entry
}

FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  "Diploma thesis" format.thesis.type output.nonnull
  school "school" output.check
  address output
  format.date "year" output.check
  new.block
  note output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.authors output
  format.title output
  howpublished output
  format.date output
  new.block
  note output
  fin.entry
  empty.misc.check
}

FUNCTION {phdthesis}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  "Dissertation" format.thesis.type output.nonnull
  school "school" output.check
  address output
  format.date "year" output.check
  new.block
  note output
  fin.entry
}

FUNCTION {proceedings}
{ output.bibitem
  editor empty$
    { organization output }
    { format.editors output.nonnull }
  if$
  format.btitle "title" output.check
  format.series output
  format.volume output
  format.number output
  organization output
  new.sentence
  format.address.publisher output
  format.date "year" output.check
  new.block
  note output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  institution "institution" output.check
  address output
  format.tr.number output.nonnull
  format.date "year" output.check
  new.block
  note output
  fin.entry
}

FUNCTION {unpublished}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  new.block
  note "note" output.check
  format.date output
  fin.entry
}

FUNCTION {default.type} { misc }

MACRO {jan} {"jan."}

MACRO {feb} {"feb."}

MACRO {mar} {"march"}

MACRO {apr} {"april"}

MACRO {may} {"may"}

MACRO {jun} {"june"}

MACRO {jul} {"july"}

MACRO {aug} {"aug."}

MACRO {sep} {"sept."}

MACRO {oct} {"oct."}

MACRO {nov} {"nov."}

MACRO {dec} {"dez."}

MACRO {acmcs} {"ACM Comp.\ Surv."}

MACRO {acta} {"Acta Inform.\"}

MACRO {cacm} {"CACM"}

MACRO {ibmjrd} {"IBM J Res.\ and Dev."}

MACRO {ibmsj} {"IBM Sys.\ J"}

MACRO {ieeeswe} {"IEEE Trans.\ on SW Eng."}

MACRO {ieeetc} {"IEEE Trans.\ on Comp."}

MACRO {ieeetcad} {"IEEE Trans.\ on CAD of Integr.\ Circuits"}

MACRO {ipl} {"Inf.\ Proc.\ Letters"}

MACRO {jacm} {"JACM"}

MACRO {jcss} {"J of Comp.\ and Sys.\ Sc.\"}

MACRO {scp} {"Sc.\ of Comp.\ Prog.\"}

MACRO {sicomp} {"SIAM J on Comp.\"}

MACRO {tocs} {"ACM Trans.\ on Comp.\ Sys.\"}

MACRO {tods} {"ACM Trans.\ on DB Sys.\"}

MACRO {tog} {"ACM Trans.\ on Graphics"}

MACRO {toms} {"ACM Trans.\ on Math.\ SW"}

MACRO {toois} {"ACM Trans.\ on Office Inf.\ Sys."}

MACRO {toplas} {"ACM TOPLAS"}

MACRO {tcs} {"Theoretical Comp.\ Sc.\"}

MACRO {fakinf} {"Univ.\ Karlsruhe, Fakult\accent127 at f. Inform."}

MACRO {it} {"Univ.\ Karlsruhe, Inst. f. Telematik"}

READ

FUNCTION {sortify}
{ purify$
  "l" change.case$
}

INTEGERS { len }

FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}

INTEGERS { et.al.char.used }

FUNCTION {initialize.et.al.char.used}
{ #0 'et.al.char.used :=
}

EXECUTE {initialize.et.al.char.used}

FUNCTION {format.lab.names}
{ 's :=
  s num.names$ 'numnames :=

  numnames #1 =
    { s #1 "{vv }{ll}" format.name$ }
    { numnames #2 =
        { s #1 "{vv }{ll}" format.name$ }
        { s #1 "{vv }{ll}" format.name$ }
      if$
    }
  if$

}

FUNCTION {author.key.label}
{ author empty$
    { key empty$

	{ cite$ #1 #3 substring$ }

	{ key }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.editor.key.label}
{ author empty$
    { editor empty$
	{ key empty$

	    { cite$ #1 #3 substring$ }

	    { key }
	  if$
	}
	{ editor format.lab.names }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.key.organization.label}
{ author empty$
    { key empty$
	{ organization empty$

	    { cite$ #1 #3 substring$ }

	    { "The " #4 organization chop.word #3 text.prefix$ }
	  if$
	}
	{ key }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {editor.key.organization.label}
{ editor empty$
    { key empty$
	{ organization empty$

	    { cite$ #1 #3 substring$ }

	    { "The " #4 organization chop.word #3 text.prefix$ }
	  if$
	}
	{ key }
      if$
    }
    { editor format.lab.names }
  if$
}

FUNCTION {calc.label}
{ type$ "book" =
  type$ "inbook" =
  or
    'author.editor.key.label
    { type$ "proceedings" =
	'editor.key.organization.label
	{ type$ "manual" =
	    'author.key.organization.label
	    'author.key.label
	  if$
	}
      if$
    }
  if$
  duplicate$

  "" swap$ * " " *
  year #3 #3 substring$ field.or.null purify$ * % CHANGED - WG - 14 Feb 1991
  'label :=
  year #3 #3 substring$ field.or.null purify$ *

  sortify 'sort.label :=
}

FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { nameptr #1 >
	{ "   " * }
	'skip$
      if$

      s nameptr "{vv{ } }{ll{ }}{  ff{ }}{  jj{ }}" format.name$ 't :=

      nameptr numnames = t "others" = and
	{ "" * }
	{ t sortify * }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "The " #4 t chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}

FUNCTION {author.sort}
{ author empty$
    { key empty$
	{ "to sort, need author or key in " cite$ * warning$
	  ""
	}
	{ key sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.editor.sort}
{ author empty$
    { editor empty$
	{ key empty$
	    { "to sort, need author, editor, or key in " cite$ * warning$
	      ""
	    }
	    { key sortify }
	  if$
	}
	{ editor sort.format.names }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.organization.sort}
{ author empty$
    { organization empty$
	{ key empty$
	    { "to sort, need author, organization, or key in " cite$ * warning$
	      ""
	    }
	    { key sortify }
	  if$
	}
	{ "The " #4 organization chop.word sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {editor.organization.sort}
{ editor empty$
    { organization empty$
	{ key empty$
	    { "to sort, need editor, organization, or key in " cite$ * warning$
	      ""
	    }
	    { key sortify }
	  if$
	}
	{ "The " #4 organization chop.word sortify }
      if$
    }
    { editor sort.format.names }
  if$
}

FUNCTION {presort}

{ calc.label
  sort.label
  "    "
  *
  type$ "book" =

  type$ "inbook" =
  or
    'author.editor.sort
    { type$ "Proceedings" =
	'editor.organization.sort
	{ type$ "handbook" =
	    'author.organization.sort
	    'author.sort
	  if$
	}
      if$
    }
  if$

  *

  "    "
  *
  year field.or.null sortify
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}

FUNCTION {search.bold_authors}
{
  type$ "boldauthor" =
  {call.type$}
  'skip$
  if$
}

ITERATE {search.bold_authors} 

ITERATE {presort}

SORT


STRINGS { longest.label last.sort.label next.extra }

INTEGERS { longest.label.width last.extra.num }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #0 int.to.chr$ 'last.sort.label :=
  "" 'next.extra :=
  #0 'longest.label.width :=
  #0 'last.extra.num :=
}

FUNCTION {forward.pass}
{ last.sort.label sort.label =
    { last.extra.num #1 + 'last.extra.num :=
      last.extra.num int.to.chr$ 'extra.label :=
    }
    { "a" chr.to.int$ 'last.extra.num :=
      "" 'extra.label :=
      sort.label 'last.sort.label :=
    }
  if$
}

FUNCTION {reverse.pass}
{ next.extra "b" =
    { "a" 'extra.label := }
    'skip$
  if$
  label extra.label * "" * 'label :=   % CHANGED - WG 14 Feb 1991
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
  extra.label 'next.extra :=
}

EXECUTE {initialize.longest.label}

ITERATE {forward.pass}

REVERSE {reverse.pass}

FUNCTION {begin.bib}
{ et.al.char.used
    { "\newcommand{\etalchar}[1]{$^{#1}$}" write$ newline$ }
    'skip$
  if$
  preamble$ empty$

    'skip$
    { preamble$ write$ newline$ }
  if$

  "\begin{thebibliography}{}" write$ newline$

}

EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}
