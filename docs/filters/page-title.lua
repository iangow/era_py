function Pandoc(document)
  local first = document.blocks[1]
  if document.meta.title or not first or first.t ~= "Header" or first.level ~= 1 then
    return document
  end

  document.meta.title = pandoc.MetaInlines(first.content)
  document.blocks:remove(1)

  local has_top_level_heading = false
  document:walk({
    Header = function(header)
      if header.level == 1 then
        has_top_level_heading = true
      end
    end
  })

  if has_top_level_heading then
    document = document:walk({
      Header = function(header)
        header.level = math.min(header.level + 1, 6)
        return header
      end
    })
  end

  if first.identifier ~= "" then
    document.blocks:insert(1, pandoc.Div({}, pandoc.Attr(first.identifier)))
  end
  return document
end
