window.onload = doStuff;

    function doStuff() {
      var table=document.getElementById("table");
      table.style.display='none';
      var loading=document.getElementById("loading");
      loading.style.display='none';

        }

    function previewFile(){
       var preview = document.querySelector('img'); //selects the query named img
       var file    = document.querySelector('input[type=file]').files[0]; //same as here
       var reader  = new FileReader();

               reader.onloadend = function () {
                   image_url = reader.result 
                   preview.src = image_url;
                   console.log(image_url)
               }

               if (file) {
                   reader.readAsDataURL(file); //reads the data as a URL
               } else {
                   preview.src = "";
               }
          }

    function classify_system(){
          table.innerHTML = "" ;
              loading.style.display='block';
              var url="classify_system?imageurl=" + image_url;
              $.get(url, function(data, status){
                table.style.display='none';
                loading.style.display='none';
                var test_data = data.results
                Highcharts.chart('piechart', {
          chart: {
                  plotBackgroundColor: null,
                  plotBorderWidth: null,
                  plotShadow: false,
                  type: 'pie'
                  },
          title: {
                  text: 'Prediction Results'
                  },
          tooltip: {
                  pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
                  },
          plotOptions: {
                    pie: {
                          allowPointSelect: true,
                          cursor: 'pointer',
                          dataLabels: {
                          enabled: true,
                          format: '<b>{point.name}</b>: {point.percentage:.1f} %',
                          style: {
                              color: (Highcharts.theme && Highcharts.theme.contrastTextColor) || 'black'
                                  }
                                      }
                            }
                          },
          series: [{
                name: 'Age Range Probability',
                colorByPoint: true,
                data: data.results
                    }]
                  })
              })
            };

       previewFile();