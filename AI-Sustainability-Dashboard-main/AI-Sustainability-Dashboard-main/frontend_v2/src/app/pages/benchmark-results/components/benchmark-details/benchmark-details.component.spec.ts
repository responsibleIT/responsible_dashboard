import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BenchmarkDetailsComponent } from './benchmark-details.component';

describe('BenchmarkResultsComponent', () => {
  let component: BenchmarkDetailsComponent;
  let fixture: ComponentFixture<BenchmarkDetailsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BenchmarkDetailsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BenchmarkDetailsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
