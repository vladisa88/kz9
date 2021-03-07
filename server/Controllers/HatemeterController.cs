using System;
using System.Diagnostics;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;

namespace Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class HatemeterController : ControllerBase
    {
        public HatemeterController()
        {
        }

        [HttpPost]
        public async Task<ActionResult<string>> Post([FromBody] string url)
        {
            try
            {
                ProcessStartInfo hatemeter = new ProcessStartInfo();
                hatemeter.FileName = "/usr/bin/python3";
                hatemeter.WorkingDirectory = "hatemeter/";
                hatemeter.Arguments = string.Format($"model.py {url}");
                hatemeter.UseShellExecute = false;
                hatemeter.RedirectStandardOutput = true;
                hatemeter.RedirectStandardError = true;
                hatemeter.CreateNoWindow = true;

                using (Process process = Process.Start(hatemeter))
                {
                    string result = await process.StandardOutput.ReadToEndAsync();
                    string error = await process.StandardError.ReadToEndAsync();
                    await process.WaitForExitAsync();
                    if (process.ExitCode == 0)
                    {
                        return Ok(result);
                    }
                    else 
                    {
                        return BadRequest(error);
                    }
                }
            }
            catch (Exception ex)
            {
                return BadRequest(ex.Message);
            }
        }
    }
}