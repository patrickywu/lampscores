options(shiny.maxRequestSize = 50*1024^2)

library(shiny)
library(shinyjs)
library(shinythemes)
library(BradleyTerry2)
library(brglm)
library(doBy)
library(qvcalc)
library(ggplot2)
library(plotly)

ui <- fluidPage(
  theme = shinytheme("cerulean"),
  
  useShinyjs(),
  
  titlePanel("Scaling the Ideology of Politicians via Bradley-Terry"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("files", "Upload pairwise comparison CSV file(s)", multiple = TRUE, accept = ".csv"),
      
      checkboxInput("include_voteview", "Include Voteview data", value = FALSE),
      
      conditionalPanel(
        condition = "input.include_voteview == true",
        fileInput("voteview_df", "Upload the corresponding voteview data (optional)", multiple = FALSE, accept = ".csv"),
        helpText("If you skipped uploading Voteview data, choose the Congress number and chamber below to automatically download the Voteview data."),
        numericInput("congress", "Congress number", value=116, min=1, max=119, step=1),
        selectInput("chamber", "Chamber", choices = c("S", "H", "HS"), selected = "S")
      ),
  
      actionButton("run", "Calculate LaMP scores", class="btn-primary")
    ),
    
    mainPanel(
      downloadButton("download_lampscore", "Download LaMP scores (CSV)"),
      downloadButton("download_lampscore_with_voteview", "Download LaMP scores (CSV) with Voteview Data"),
      checkboxInput("flip_plot", "Show DW-NOMINATE v. LaMP scores plot", value=FALSE),
      
      conditionalPanel(
        condition = "input.flip_plot == false",
        plotOutput("abilities_plot", width="800px", height = "700px")
      ),
      
      conditionalPanel(
        condition = "input.flip_plot == true",
        plotlyOutput("dw_plot", width="800px", height="600px")
      )
    )
  )
)

server <- function(input, output, session) {
  
  shinyjs::hide("download_lampscore")
  shinyjs::hide("download_lampscore_with_voteview")
  
  observe({
    if (isTruthy(lampscore_df())) {
      shinyjs::show("download_lampscore")
    } else {
      shinyjs::hide("download_lampscore")
    }
  })
  
  # Read and combine uploaded data
  uploaded_pairwise_data <- reactive({
    req(input$files)
    dfs <- lapply(input$files$datapath, read.csv, stringsAsFactors = FALSE)
    do.call(rbind, dfs)
  })
  
  # Read the Voteview data, if available
  voteview_df <- reactive({
    if (!isTRUE(input$include_voteview)){
      return(NULL)
    }
    
    if(!isTruthy(input$voteview_df)){
      voteview_df_url <- paste0("https://voteview.com/static/data/out/members/", 
                                input$chamber,
                                input$congress,
                                "_members.csv")
      voteview_data <- read.csv(voteview_df_url)
    }
    else{
      voteview_data <- read.csv(input$voteview_df$datapath)
    }
    return(voteview_data)
  })
  
  # Fit model after user clicks analysis button
  comp_results <- eventReactive(input$run, {
    df <- uploaded_pairwise_data()
    
    name_bioguide_id <- data.frame(
      bioguide_id = c(df$bioguide_id0, df$bioguide_id1),
      bioname_canonical = c(df$name0, df$name1),
      party_abbrev = c(df$party0, df$party1),
      stringsAsFactors = FALSE
    )
    
    name_bioguide_id <- name_bioguide_id[!duplicated(name_bioguide_id$bioguide_id), ]
    row.names(name_bioguide_id) <- NULL
    
    # Collapse duplicate match‑ups (sum wins)
    collapse_df <- summaryBy(win0 + win1 ~ bioguide_id0 + bioguide_id1, FUN = sum, data = df)
    colnames(collapse_df) <- c("bioguide_id0", "bioguide_id1", "win0", "win1")
    
    # Factorize competitors with a common level set
    collapse_df$id0 <- factor(collapse_df$bioguide_id0, levels = unique(c(collapse_df$bioguide_id0, collapse_df$bioguide_id1)))
    collapse_df$id1 <- factor(collapse_df$bioguide_id1, levels = unique(c(collapse_df$bioguide_id0, collapse_df$bioguide_id1)))
    
    # Choose a random person to be the reference category --- it doesn't matter, because we will rescale to 0-1
    ch <- sort(unique(c(uploaded_pairwise_data()$bioguide_id0, uploaded_pairwise_data()$bioguide_id1)))
    
    # Fit bias‑reduced Bradley–Terry model
    mod <- BTm(outcome = cbind(win0, win1), id0, id1, ~id, id="id", data=collapse_df, refcat=sample(ch, 1), br=TRUE)
    
    list(model = mod, name_bioguide_id = name_bioguide_id)
  })
  
  # Compute scaled LaMP scores with quasi‑SEs
  lampscore_df <- reactive({
    req(comp_results())
    mod <- comp_results()$model
    
    abilities <- BTabilities(mod)
    qv <- qvcalc(abilities)

    rng  <- range(abilities[, 1])
    scaled <- (abilities[, 1] - rng[1]) / diff(rng)
    qse <- qv$qvframe[, 3] / diff(rng)
    
    df <- data.frame(
      bioguide_id = rownames(abilities),
      lampscore = scaled,
      qse = qse,
      stringsAsFactors = FALSE
    )
    
    final_df <- merge(df, comp_results()$name_bioguide_id, by="bioguide_id", all.x=TRUE)
    
    # re-order into a logical order
    final_df <- final_df[c("bioguide_id", "bioname_canonical", "party_abbrev", "lampscore", "qse")]
    
    return(final_df)
  })

  output$download_lampscore <- downloadHandler(
    filename = function() {
      paste0("lampscore_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(lampscore_df(), file, row.names = FALSE)
    }
  )
  
  observe({
    if (isTruthy(lampscore_with_voteview_df())){
      shinyjs::show("download_lampscore_with_voteview")
    } else{
      shinyjs::hide("download_lampscore_with_voteview")
    }
  })
  
  observe({
    if (isTruthy(lampscore_with_voteview_df())){
      shinyjs::show("flip_plot")
    } else{
      shinyjs::hide("flip_plot")
    }
  })
  
  lampscore_with_voteview_df <- reactive({
    if(!isTRUE(input$include_voteview)){
      return(NULL)
    }
    
    req(lampscore_df(), voteview_df())
    ls_df <- lampscore_df()
    vv_df <- voteview_df()
    lampscore_with_voteview <- merge(ls_df, vv_df, by="bioguide_id", all.x=TRUE)
    return(lampscore_with_voteview)
  })
  
  output$download_lampscore_with_voteview <- downloadHandler(
    filename = function(){
      paste0("lampscore_with_voteview_", Sys.Date(), ".csv")
    },
    content = function(file){
      write.csv(lampscore_with_voteview_df(), file, row.names = FALSE)
    }
  )
  
  # Plot - only LaMP scores
  output$abilities_plot <- renderPlot({
    req(lampscore_df())
    
    df <- lampscore_df()
    
    df$party <- ifelse(df$party_abbrev == "D", "Democratic", ifelse(df$party_abbrev == "R", "Republican", "Independent"))
    
    ggplot(df, aes(x = reorder(bioname_canonical, lampscore), y = lampscore, color = party)) +
      geom_point(size = 2) +
      geom_errorbar(aes(ymin = lampscore - 1.96*qse, ymax = lampscore + 1.96*qse), width = 0.2) +
      scale_color_manual(values = c("Democratic"="#2E74C0", "Republican"="#CB454A", "Independent"="#3C9D32"), breaks=c("Democratic", "Independent", "Republican"), name="Party") +
      coord_flip() +
      labs(x = NULL, y = "LaMP scores with 95% confidence intervals") +
      theme_minimal()
  })
  
  # DW-NOMINATE vs. LaMP scores
  output$dw_plot <- renderPlotly({
    
    if (!isTRUE(input$include_voteview)){
      return(NULL)
    }
    
    req(lampscore_with_voteview_df())
    df <- lampscore_with_voteview_df()

    df$party <- ifelse(df$party_abbrev == "D", "Democratic", ifelse(df$party_abbrev == "R", "Republican", "Independent"))
    
    p <- plot_ly(
      data = df,
      x = ~nominate_dim1,
      y = ~lampscore,
      type = "scatter",
      mode = "markers",
      color = ~party,
      colors = c("Republican" = "#CB454A", "Democratic" = "#2E74C0", "Independent" = "#3C9D32"),
      text   = ~paste0(bioname_canonical, " (", party_abbrev, "-", state_abbrev, ")", "<br>LaMP Score: ", round(lampscore, 3), "<br>DW-NOMINATE: ", round(nominate_dim1, 3)),
      hoverinfo = "text"
    )
    
    p <- layout(
      p,
      xaxis = list(title = "DW-NOMINATE", showgrid=FALSE, zeroline=FALSE, showline=TRUE),
      yaxis = list(title = "LaMP score", showgrid=FALSE, zeroline=FALSE, showline=TRUE),
      legend = list(title = list(text = "Party"))
    )
    
    p 
  })
}

shinyApp(ui, server)